from time import time
from collections import deque
import numpy as np
from random import shuffle

from classes.node import Node
from classes.arc import Arc
from classes.decision_diagram import DecisionDiagram


class BISPCompilerPessimistic:
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
        root_node = Node(0, 0, [list(diagram.var_order["follower"]), 0])
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

        # Queues of nodes
        current_layer_queue = deque()

        # Create follower layers
        for layer in range(instance.nF):
            var_index = diagram.var_order["follower"][layer]
            for node in diagram.graph_map[layer].values():
                    current_layer_queue.append(node)
            
            # Width limit
            if len(current_layer_queue) > diagram.max_width:
                current_layer_queue = self.reduce_queue(instance, diagram.reduce_method, current_layer_queue, diagram.max_width)

            self.logger.debug("Follower layer {} -> Variable index = {}, Queue size = {}".format(
                layer, var_index, len(current_layer_queue)
            ))

            # Compile new layer
            while len(current_layer_queue):
                node = current_layer_queue.popleft()
                zero_head = self.create_follower_node(instance, layer + 1, var_index, node, 0, diagram.problem_setting)
                one_head = self.create_follower_node(instance, layer + 1, var_index, node, 1, diagram.problem_setting)
                zero_head_feasibility = self.check_completion_bounds(instance, completion_bounds, zero_head)\
                                        and instance.known_y_values.get(var_index) != 1
                one_head_feasibility = self.check_completion_bounds(instance, completion_bounds, one_head)\
                                        and instance.known_y_values.get(var_index) != 0

                # Zero head
                if zero_head_feasibility:
                    zero_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if zero_head.hash_key in diagram.graph_map[zero_head.layer]:
                        found_node = diagram.graph_map[zero_head.layer][zero_head.hash_key]
                        self.update_costs(node=found_node, new_node=zero_head)
                        zero_head = found_node
                        # Create arc if node was not already connected
                        if node.hash_key not in diagram.graph_map[node.layer]:
                            arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                    else:
                        diagram.add_node(zero_head)  # TODO: add nodes after reducing next_layer_queue
                        # Create arc
                        arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
                        diagram.add_arc(arc)
                        next_layer_queue.append(zero_head)
                
                # One head
                if one_head_feasibility:
                    one_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if one_head.hash_key in diagram.graph_map[one_head.layer]:
                        found_node = diagram.graph_map[one_head.layer][one_head.hash_key]
                        self.update_costs(node=found_node, new_node=one_head)
                        one_head = found_node
                        if node.hash_key not in diagram.graph_map[node.layer]:
                            # Create arc if node was not already connected
                            arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                    else:
                        diagram.add_node(one_head)  # TODO: add nodes after reducing next_layer_queue
                        # Create arc
                        arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                        diagram.add_arc(arc)
                        next_layer_queue.append(one_head)

            # Width limit
            if len(next_layer_queue) > diagram.max_width:
                next_layer_queue = self.reduce_queue(instance, diagram.reduce_method, next_layer_queue, diagram.max_width)

            # Update queues
            current_layer_queue = next_layer_queue
            next_layer_queue = deque()
        
        # Compress follower layers
        if diagram.encoding == "compact":
            root_node = diagram.root_node
            sink_node = diagram.sink_node
            diagram.arcs = list()
            diagram.nodes = [root_node, sink_node] + list(current_layer_queue)
            root_node.outgoing_arcs = list()
            for node in current_layer_queue:
                node.incoming_arcs = list()
                arc = Arc(root_node, node, value=1, cost=node.follower_cost, var_index=None, player="follower")
                diagram.add_arc(arc)

    def compile_leader_layers(self, diagram, instance):
        self.logger.debug("Compiling leader layers")

        if diagram.encoding == "compact":
            # Compile compressed leader layer
            current_layer_queue = deque(diagram.graph_map[instance.nF].values())
            while len(current_layer_queue):
                node = current_layer_queue.popleft()
                arc = Arc(tail=node, head=diagram.sink_node, value=0, cost=0, var_index=None, player="leader")
                diagram.add_arc(arc)
        
        elif diagram.encoding == "extended":
            raise ValueError("Extended encoding for general instances not implemented yet.")

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
            # Select follower variables with known value
            order["follower"] = [j for j in instance.known_y_values]
            remaining_columns = [j for j in range(instance.nF) if j not in instance.known_y_values]

            # Select follower variable with more appearances in D together with the already included variables
            degree_sequence = {j: 0 for j in remaining_columns}
            for i in range(instance.mF):
                for j in remaining_columns:
                    if instance.D[i][j] != 0 and instance.interaction[i] == "follower": 
                        degree_sequence[j] += 1
                        for k in instance.known_y_values:
                            if instance.D[i][k] != 0:
                                degree_sequence[j] += 1
            order["follower"].append(max(degree_sequence.items(), key=lambda x: x[1])[0])
            remaining_columns = [i for i in range(instance.nF) if i != order["follower"][-1]]

            # Select follower variable with the most appearances in D where the last column also appears
            while remaining_columns:
                degree_sequence = {j: [None, 0, 0] for j in remaining_columns}  # [idx, joint appearances, degree]
                last_column = order["follower"][-1]
                for i in range(instance.mF):
                    for idx, j in enumerate(remaining_columns):
                        degree_sequence[j][0] = idx
                        if instance.D[i][j] != 0 and instance.interaction[i] == "follower":  # Only follower constrs
                            degree_sequence[j][2] += 1  # Degree
                            if instance.D[i][last_column] != 0:
                                degree_sequence[j][1] += 1  # Joint appearance
                column = max(degree_sequence.items(), key=lambda x: (x[1][1], x[1][2]))
                order["follower"].append(column[0])
                remaining_columns.pop(column[1][0])

        else:
            raise ValueError("Invalid ordering heuristic value")
        
        order["leader"] = [i for i in range(instance.nL)]

        return order, time() - t0
    
    def create_follower_node(self, instance, layer, var_index, parent_node, value, problem_setting):
        if value == 0:
            node = Node(None, layer, np.copy(parent_node.state), 0)
            node.follower_cost = float(parent_node.follower_cost)
            node.leader_cost = float(parent_node.leader_cost)
        else:
            node = Node(None, layer, parent_node.state + instance.D[:, var_index], 1)
            node.follower_cost = parent_node.follower_cost + instance.d[var_index]
            node.leader_cost = float(parent_node.leader_cost)

        return node
    
    def update_costs(self, node, new_node):
        node.follower_cost = min(node.follower_cost, new_node.follower_cost)
        node.leader_cost = min(node.leader_cost, new_node.leader_cost)

    def reduce_queue(self, instance, reduce_method, queue, max_width):
        def check_diversity_criterion(instance, nodes, new_node):
            for node in nodes:
                if blocking_distance(instance, new_node, node) <= 0:
                    return False
            
            return True
        
        def blocking_distance(instance, node_1, node_2):
            distance = 0
            for i in range(instance.mF):
                if instance.interaction[i] == "both":
                    if node_1.state[i] < node_2.state[i]:
                        distance += 1

            return distance
        
        if reduce_method == "follower_cost":
            sorted_queue = deque(sorted(queue, key=lambda x: int(0.9 * x.follower_cost + 0.1 * x.leader_cost)))
        elif reduce_method == "minmax_state":
            sorted_queue = deque(sorted(queue, key=lambda x: (max(x.state - instance.b), sum(x.state), x.follower_cost)))
        elif reduce_method == "random":
            sorted_queue = queue
            shuffle(sorted_queue)
        elif reduce_method == "minsum_state":
            sorted_queue = deque(sorted(queue, key=lambda x: sum(x.state), reverse=True))

        filtered_queue = deque()
        remaining_nodes = deque()
        
        # Check diversity of the queue
        for node in sorted_queue:
            if check_diversity_criterion(instance, filtered_queue, node) and len(filtered_queue) < max_width:
                filtered_queue.append(node)
            else:
                remaining_nodes.append(node)
        
        # Replenish queue if too small
        while len(filtered_queue) < max_width:
            filtered_queue.append(remaining_nodes.popleft())
        
        return filtered_queue
    
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