from time import time
from collections import deque
import numpy as np
from random import shuffle

from classes.node import Node
from classes.arc import Arc
from classes.decision_diagram import DecisionDiagram


class BISPOptimisticCompiler:
    def __init__(self, logger):
        self.logger = logger

    def compile(self, instance, args):
        """
            This method compiles a DD for BISP-KC instances, starting with the follower layers 
            and continuing with the leader layers
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int), ordering_heuristic (str), [optional] Y (list)
            Returns: diagram (class DecisionDiagram)

            TODO: 
                - Check when state component will not change anymore
        """

        t0 = time()
        self.logger.info("Compilation started")

        graph_map = dict()
        if args["encoding"] == "compact":
            graph_map = {l: dict() for l in range(len(instance.graph.nodes) + 2)}
        elif args["encoding"] == "extended":
            graph_map = {l: dict() for l in range(2 * len(instance.graph.nodes) + 2)}

        # Create diagram
        diagram = DecisionDiagram(0, args)
        diagram.var_order, diagram.ordering_heuristic_runtime = self.get_ordering_heuristic(instance, args["ordering_heuristic"])
        diagram.graph_map = graph_map
        root_node = Node(
            0, 
            0, 
            np.array(list(diagram.var_order["follower"]) + [0]),  # State variable: Node list + LHS
            0,
            0
        )
        diagram.add_node(root_node)
        if args["encoding"] == "compact":
            sink_node = Node(-1, len(instance.graph.nodes) + 1)
        elif args["encoding"] == "extended":
            sink_node = Node(-1, 2 * len(instance.graph.nodes) + 1)
        diagram.add_node(sink_node)

        # Compile follower layers
        self.compile_follower_layers(diagram, instance)
        
        # Compile leader layers
        self.compile_leader_layers(diagram, instance)

        # Clean diagram
        self.clean_diagram(diagram)

        # Update diagram data
        diagram.compilation_runtime = time() - t0
        
        self.logger.info("Diagram succesfully compiled -> Time elapsed = {}s, Node count = {}, Arc count = {}, Width = {}".format(
            round(diagram.compilation_runtime), diagram.node_count, diagram.arc_count, diagram.width
        ))

        return diagram
    
    def compile_follower_layers(self, diagram, instance):
        self.logger.debug("Compiling follower layers")

        var_idx_to_layer = {idx: layer for layer, idx in enumerate(diagram.var_order["follower"])}

        # Compute completion bounds for follower constrs
        completion_bounds = self.initialize_follower_completion_bounds(instance)

        # Create follower layers
        for layer in range(instance.nF):
            var_index = diagram.var_order["follower"][layer]
            current_layer_queue = deque(diagram.graph_map[layer].values())

            # Update completion bound
            self.update_follower_completion_bounds(instance, completion_bounds, var_index)

            # Width limit
            if len(current_layer_queue) > diagram.max_width:
                current_layer_queue = self.reduce_queue(diagram.reduce_method, current_layer_queue, diagram.max_width)
                # Empty and update graph map
                diagram.graph_map[layer] = dict()
                for node in current_layer_queue:
                    diagram.graph_map[layer][node.hash_key] = node

            self.logger.debug("Follower layer {} -> Variable index = {}, Queue size = {}".format(
                layer, var_index, len(current_layer_queue)
            ))

            # Compile new layer
            while len(current_layer_queue):
                node = current_layer_queue.popleft()
                zero_head = self.create_follower_node(instance, node, var_idx_to_layer, 0)
                one_head = self.create_follower_node(instance, node, var_idx_to_layer, 1)
                child_nodes = [
                    (zero_head, self.check_completion_bounds(instance, completion_bounds, zero_head)),
                    (one_head, self.check_completion_bounds(instance, completion_bounds, one_head))
                ]

                for child_node, feasibility in child_nodes:
                    if feasibility:
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map[child_node.layer]:
                            new_node = diagram.graph_map[child_node.layer][child_node.hash_key]
                            self.update_costs(new_node, child_node)
                        else:
                            new_node = child_node
                            new_node.id = diagram.node_count
                            diagram.add_node(child_node)

                        arc = Arc(
                            node, 
                            new_node, 
                            child_node.type, 
                            instance.d[var_index] * child_node.type, 
                            instance.cF[var_index] * child_node.type, 
                            None, 
                            "follower"
                        )
                        diagram.add_arc(arc)

        # Width limit for last layer
        current_layer_queue = deque(diagram.graph_map[instance.nF].values())
        if len(current_layer_queue) > diagram.max_width:
            current_layer_queue = self.reduce_queue(diagram.reduce_method, current_layer_queue, diagram.max_width)
            # Empty and update graph map
            diagram.graph_map[instance.nF] = dict()
            for node in current_layer_queue:
                diagram.graph_map[instance.nF][node.hash_key] = node

        self.logger.debug("Follower layer {} -> Variable index = {}, Queue size = {}".format(
            instance.nF, var_index, len(current_layer_queue)
        ))
        
        # Compress follower layers
        if diagram.encoding == "compact":
            root_node = diagram.root_node
            sink_node = diagram.sink_node
            diagram.arcs = list()
            diagram.nodes = [root_node, sink_node] + list(current_layer_queue)
            root_node.outgoing_arcs = list()
            for node in current_layer_queue:
                node.incoming_arcs = list()
                arc = Arc(root_node, node, None, node.follower_cost, 0, None, "follower")
                diagram.add_arc(arc)

    def compile_leader_layers(self, diagram, instance):
        self.logger.debug("Compiling leader layers")

        current_layer_queue = deque([node for node in diagram.graph_map[instance.nF].values()])

        if diagram.encoding == "compact":
            # Compile compressed leader layer
            while len(current_layer_queue):
                node = current_layer_queue.popleft()
                arc = Arc(node, diagram.sink_node, None, 0, 0, None, "leader")
                diagram.add_arc(arc)
        
        elif diagram.encoding == "extended":
            raise ValueError("Extended encoding for BISP-KC instances not implemented yet.")

    def get_ordering_heuristic(self, instance, ordering_heuristic):
        """
            This method retrieves a variable ordering.

            Args: instance (class Instance)
            Returns: dict of sorted indices (dict)
        """
        t0 = time()
        order = {"leader": list(), "follower": list()}

        # Sum coeffs of the left hand side
        if ordering_heuristic == "lhs_coeffs":
            self.logger.debug("Variable ordering heuristic: LHS coeffs")
            for j in range(instance.nF):
                coeffs_sum = sum(instance.D[i][j] for i in range(instance.mF) if instance.interaction[i] == "both")
                order["follower"].append((j, coeffs_sum))
            order["follower"].sort(key=lambda x: x[1])  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]

        # Leader cost
        elif ordering_heuristic == "leader_cost":
            self.logger.debug("Variable ordering heuristic: leader cost")
            for j in range(instance.nF):
                order["follower"].append((j, instance.cF[j]))
            order["follower"].sort(key=lambda x: x[1])  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]

        # Follower costs
        elif ordering_heuristic == "follower_cost":
            self.logger.debug("Variable ordering heuristic: follower cost")
            order["follower"] = sorted([j for j in range(instance.nF)], key=lambda x: instance.d[x])

        # Leader feasibility
        elif ordering_heuristic == "leader_feasibility":
            self.logger.debug("Variable ordering heuristic: leader feasibility")
            for j in range(instance.nF):
                order["follower"].append((j, max([instance.D[i][j] for i in range(instance.mF)]), sum([instance.D[i][j] for i in range(instance.mF)])))
            order["follower"].sort(key=lambda x: (x[1], x[2]))  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]
                
        # Lexicographic
        elif ordering_heuristic == "lexicographic":
            self.logger.debug("Variable ordering heuristic: lexicographic")
            order["follower"] = [i for i in range(instance.nF)]

        # Max-connected-degree
        elif ordering_heuristic == "max_connected_degree":
            self.logger.debug("Variable ordering heuristic: max-connected-degree")
            columns = [j for j in range(instance.nF)]

            # Select max degree node
            max_degree = 0
            for j in range(instance.nF):
                degree = len(instance.graph.edges(j))
                if degree > max_degree:
                    node = j
                    max_degree = degree
            var_order = [columns.pop(node)]

            # Select connected node with max degree
            while columns:
                max_total_degree = -float("inf")
                max_degree = -float("inf")
                node = None
                node_idx = None
                for idx, j in enumerate(columns):
                    degree = len([v for v in var_order if instance.graph.has_edge(j, v)])
                    total_degree = len(instance.graph.edges(j))
                    if degree > max_degree:
                        max_degree = degree
                        max_total_degree = total_degree
                        node = j
                        node_idx = idx
                    elif degree == max_degree and total_degree > max_total_degree:
                        max_degree = degree
                        max_total_degree = total_degree
                        node = j
                        node_idx = idx
                var_order.append(columns.pop(node_idx))
            order["follower"] = var_order

        else:
            raise ValueError("Invalid ordering heuristic value")
        
        order["leader"] = [i for i in range(instance.nL)]

        return order, time() - t0
    
    def initialize_follower_completion_bounds(self, instance):
        completion_bounds = sum(min(0, instance.C[0][j]) for j in range(instance.nL))
        completion_bounds += sum(min(0, instance.D[0][j]) for j in range(instance.nF))
        
        return completion_bounds

    def update_follower_completion_bounds(self, instance, completion_bounds, var_index):
        completion_bounds -= min(0, instance.D[0, var_index])

    def check_completion_bounds(self, instance, completion_bounds, node):
        return node.state[-1] + completion_bounds <= instance.b[0]

    def create_follower_node(self, instance, parent_node, var_idx_to_layer, value):
        child_node = None
        vertex = None
        if parent_node.state.size > 1:
            vertex = parent_node.state[:-1][0]
            
        # Node with follower value 1
        if value == 1:
            if vertex != None:
                child_node_state = np.array([v for v in parent_node.state[1:-1] if not instance.graph.has_edge(vertex, v)])
                if child_node_state.size > 0:
                    child_node = Node(
                        None, 
                        var_idx_to_layer[child_node_state[0]], 
                        np.append(child_node_state, parent_node.state[-1:] + instance.D[0][vertex]),  # Keep [-1:] to preserve data type (ndarray)
                        parent_node.follower_cost + instance.d[vertex],
                        parent_node.leader_cost + instance.cF[vertex],
                        1
                    )
                else:
                    child_node = Node(
                        None, 
                        len(instance.graph.nodes), 
                        parent_node.state[-1:] + instance.D[0][vertex],  # Keep [-1:] to preserve data type (ndarray)
                        parent_node.follower_cost + instance.d[vertex],
                        parent_node.leader_cost + instance.cF[vertex],
                        1
                    )
            else:
                child_node = Node(None, parent_node.layer + 1)

        # Node with follower value 0
        else:
            if vertex != None:
                child_node_state = parent_node.state[1:-1]
                if child_node_state.size > 0:
                    child_node = Node(
                        None, 
                        var_idx_to_layer[child_node_state[0]], 
                        np.append(child_node_state, parent_node.state[-1:]),  # Keep [-1:] to preserve data type (ndarray)
                        parent_node.follower_cost,
                        parent_node.leader_cost,
                        0
                    )
                else:
                    child_node = Node(
                        None, 
                        len(instance.graph.nodes), 
                        parent_node.state[-1:],  # Keep [-1:] to preserve data type (ndarray)
                        parent_node.follower_cost,
                        parent_node.leader_cost,
                        0
                    )
                
        return child_node
    
    def update_costs(self, node, new_node):
        node.follower_cost = min(node.follower_cost, new_node.follower_cost)
        node.leader_cost = min(node.leader_cost, new_node.leader_cost)

    def reduce_queue(self, reduce_method, queue, max_width):
        if reduce_method == "follower_cost":
            reduced_queue = deque(sorted(queue, key=lambda x: x.follower_cost)[:max_width])

        elif reduce_method == "random":
            sorted_queue = list(queue)
            shuffle(sorted_queue)
            reduced_queue = deque(sorted_queue[:max_width])
        
        else:
            raise ValueError("Invalid ordering heuristic value")
        
        return reduced_queue
    
    def clean_diagram(self, diagram):
        t0 = time()
        self.logger.debug("Executing bottom-up filtering to remove unreachable nodes")

        self.bottom_up_filtering(diagram)

        # Relabel nodes (Not required)
        label = 1
        for layer in range(1, diagram.sink_node.layer):
            for node in diagram.graph_map[layer].values():
                node.id = label
                label += 1

        self.logger.debug("Bottom-up filtering done -> Time elapsed = {}s".format(round(time() - t0)))

    def bottom_up_filtering(self, diagram):
        for node in diagram.nodes:
            node.outgoing_arcs = list()
        diagram.nodes = list()
        diagram.arcs = list()
        diagram.graph_map = {layer: dict() for layer in range(diagram.sink_node.layer + 1)}
        diagram.add_node(diagram.root_node)
        diagram.add_node(diagram.sink_node)
        queue = deque([diagram.sink_node])
        while queue:
            node = queue.popleft()
            for arc in node.incoming_arcs:
                if arc.tail.hash_key not in diagram.graph_map[arc.tail.layer]:
                    diagram.add_node(arc.tail)
                    queue.append(arc.tail)
                diagram.add_arc(arc, update_in_outgoing_arcs=False)
                arc.tail.outgoing_arcs.append(arc)