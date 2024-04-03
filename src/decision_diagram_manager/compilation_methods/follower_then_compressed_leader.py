from time import time
from collections import deque
import numpy as np

from . import constants

from classes.node import Node
from classes.arc import Arc
from decision_diagram_manager.operations import Operations

from algorithms.utils.solve_follower_problem import solve as solve_follower_problem


class FollowerThenCompressedLeaderCompiler:
    def __init__(self, logger):
        self.logger = logger
        self.operations = Operations(logger)

    def compile(self, diagram, instance, max_width, ordering_heuristic, discard_method, Y, skip_brute_force_compilation):
        """
            This method compiles a DD, starting with the follower and continuing with a single compressed leader layer.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int), ordering_heuristic (str), [optional] Y (list)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram -> Compilation method: follower_then_compressed_leader - MaxWidth: {} - OrderingHeuristic: {} - DiscardMethod: {}".format(max_width, ordering_heuristic, discard_method))
        var_order = self.operations.ordering_heuristic(instance, ordering_heuristic, compressed_leader=True)

        diagram.max_width = max_width
        diagram.ordering_heuristic = ordering_heuristic
        diagram.var_order = var_order
        diagram.graph_map = {l: dict() for l in range(instance.Fcols + 2)}

        # Create root and sink nodes
        root_node = Node(id="root", layer=0, state=np.zeros(instance.Frows))
        for i in range(instance.Frows):
            if instance.interaction[i] == "leader":
                root_node.state[i] = -float("inf")
        diagram.add_node(root_node)
        sink_node = Node(id="sink", layer=instance.Fcols + 1)
        diagram.add_node(sink_node)

        # Create dummy long arc
        self.logger.debug("Solving follower problem to build dummy arc")
        M = 1e6 #solve_follower_problem(instance, sense="maximize")[0]
        self.logger.debug("Big-M value: {}".format(M))
        dummy_arc = Arc(tail=root_node, head=sink_node, value=0, cost=M, var_index=-1, player=None)
        diagram.add_arc(dummy_arc)

        ## Brute force compilation
        if not skip_brute_force_compilation:
            self.brute_force_compilation(instance, diagram, var_order, max_width, discard_method)

        ## Sampled solutions compilation
        if Y:
            self.sampled_solutions_compilation(instance, diagram, var_order, max_width, Y)

        # ## Single solution-based compilation (last resort)
        # if diagram.width == 1 and len(Y) == 1:
        #     self.logger.debug("Using HPR based branching")
        #     self.HPR_based_compilation(instance, diagram, var_order, max_width, discard_method)

        # Update diagram data
        diagram.compilation_method = "follower_then_compressed_leader"
        diagram.compilation_runtime = time() - t0

        # # Reduce diagram algorithm
        # self.operations.reduce_diagram(diagram)
        
        self.logger.info("Diagram succesfully compiled -> Time elapsed: {} s - Node count: {} - Arc count: {} - Width: {}".format(
            diagram.compilation_runtime, diagram.node_count + 2, diagram.arc_count, diagram.width
        ))

        return diagram
    
    def brute_force_compilation(self, instance, diagram, var_order, max_width, discard_method):
        ##### Compile y-solutions by binary branching
        self.logger.debug("Compiling new solutions")

        # Queues of nodes
        current_layer_queue = deque()
        next_layer_queue = deque()

        # Compute completion bounds for follower constrs
        completion_bounds = self.initialize_completion_bounds(instance)

        # Create new nodes and arcs
        fixed_y_values = {j: False for j in range(instance.Fcols)}
        current_layer_queue.append(diagram.root_node)
        for layer in range(instance.Fcols + 1):
            player = "follower"
            # Choose player
            if layer >= instance.Fcols:
                player = "leader"
                var_index = var_order[player][layer - instance.Fcols]
            else:
                var_index = var_order[player][layer]
            self.logger.debug("{} layer: {} - Variable index: {} - Queue size: {}".format(
                player.capitalize(), layer, var_index, len(current_layer_queue)
            ))

            # Update completion bound
            self.operations.update_completions_bounds(instance, completion_bounds, var_index, player)

            # Compile follower layers
            if player == "follower":
                while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
                    node = current_layer_queue.popleft()
                    zero_head = self.operations.create_zero_node(layer + 1, parent_node=node)
                    one_head = self.operations.create_one_node(instance, layer + 1, var_index, parent_node=node, player=player)

                    # # Remove fixed state components
                    # self.remove_fixed_state_component(instance, fixed_y_values, [zero_head, one_head])

                    # Zero head
                    if self.operations.check_completion_bounds(instance, completion_bounds, zero_head) and instance.known_y_values.get(var_index) != 1:
                        zero_head.id = diagram.node_count + 1
                        # Check if node was already created
                        if zero_head.hash_key in diagram.graph_map[zero_head.layer]:
                            found_node = diagram.graph_map[zero_head.layer][zero_head.hash_key]
                            self.operations.update_costs(node=found_node, new_node=zero_head)
                            zero_head = found_node
                            if node.hash_key not in diagram.graph_map[node.layer]:
                                # Create arc
                                arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(zero_head)  # TODO: add nodes after reducing next_layer_queue
                            # Create arc
                            arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                            next_layer_queue.append(zero_head)
                    
                    # One head
                    if self.operations.check_completion_bounds(instance, completion_bounds, one_head) and instance.known_y_values.get(var_index) != 0:
                        one_head.id = diagram.node_count + 1
                        # Check if node was already created
                        if one_head.hash_key in diagram.graph_map[one_head.layer]:
                            found_node = diagram.graph_map[one_head.layer][one_head.hash_key]
                            self.operations.update_costs(node=found_node, new_node=one_head)
                            one_head = found_node
                            if node.hash_key not in diagram.graph_map[node.layer]:
                                # Create arc
                                arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(one_head)  # TODO: add nodes after reducing next_layer_queue
                            # Create arc
                            arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                            next_layer_queue.append(one_head)

                fixed_y_values[var_index] = True

            # Compile compressed leader layer
            else:
                while len(current_layer_queue):
                    node = current_layer_queue.popleft()
                    arc = Arc(tail=node, head=diagram.sink_node, value=0, cost=0, var_index=-1, player="leader")
                    for i in range(instance.Frows):
                        if instance.interaction[i] == "both":
                            arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
                    diagram.add_arc(arc)

            if diagram.node_count >= constants.MAX_NODE_COUNT:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if len(next_layer_queue) > max_width:
                next_layer_queue = self.operations.reduce_queue(instance, discard_method, next_layer_queue, max_width, player)

            # Update queues
            current_layer_queue = next_layer_queue
            next_layer_queue = deque()

        # Clean diagram
        self.operations.clean_diagram(diagram)

    def sampled_solutions_compilation(self, instance, diagram, var_order, max_width, Y):
        self.logger.debug("Compiling y-solutions in sample set Y ({})".format(len(Y)))
        for idx, y in enumerate(Y):
            # if diagram.width <= max_width:
            self.logger.debug("Compiling sampled solution {}/{}".format(idx + 1, len(Y)))
            fixed_y_values = {j: False for j in range(instance.Fcols)}
            child_node = diagram.root_node

            for layer in range(instance.Fcols):
                # Choose player
                if layer >= instance.Fcols:
                    player = "leader"
                    var_index = var_order[player][layer - instance.Fcols]
                else:
                    player = "follower"
                    var_index = var_order[player][layer]
                var_index = var_order["follower"][layer]

                # Create nodes
                if diagram.node_count < constants.MAX_NODE_COUNT:
                    node = child_node
                    if y[var_index] == 0:
                        child_node = self.operations.create_zero_node(layer + 1, node)
                    else:
                        child_node = self.operations.create_one_node(instance, layer + 1, var_index, node, player="follower")

                    # # Remove fixed state components
                    # self.remove_fixed_state_component(instance, fixed_y_values, [child_node])

                    # Zero head
                    if y[var_index] == 0:
                        child_node.id = diagram.node_count + 1
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map[child_node.layer]:
                            found_node = diagram.graph_map[child_node.layer][child_node.hash_key]
                            self.operations.update_costs(node=found_node, new_node=child_node)
                            child_node = found_node
                            if node.hash_key not in diagram.graph_map[node.layer]:
                                # Create arc
                                arc = Arc(tail=node, head=child_node, value=0, cost=0, var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(child_node)
                            # Create arc
                            arc = Arc(tail=node, head=child_node, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)

                            # Compile compressed leader layer
                            if layer == instance.Fcols - 1:
                                arc = Arc(tail=child_node, head=diagram.sink_node, value=0, cost=0, var_index=-1, player="leader")
                                for i in range(instance.Frows):
                                    if instance.interaction[i] == "both":
                                        arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
                                diagram.add_arc(arc)
                    
                    # One head
                    else:
                        child_node.id = diagram.node_count + 1
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map[child_node.layer]:
                            found_node = diagram.graph_map[child_node.layer][child_node.hash_key]
                            self.operations.update_costs(node=found_node, new_node=child_node)
                            child_node = found_node
                            if node.hash_key not in diagram.graph_map[node.layer]:
                                # Create arc
                                arc = Arc(tail=node, head=child_node, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(child_node)
                            # Create arc
                            arc = Arc(tail=node, head=child_node, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)

                            # Compile compressed leader layer
                            if layer == instance.Fcols - 1:
                                arc = Arc(tail=child_node, head=diagram.sink_node, value=0, cost=0, var_index=-1, player="leader")
                                for i in range(instance.Frows):
                                    if instance.interaction[i] == "both":
                                        arc.block_values[i] = instance.b[i] - child_node.state[i] + 1 
                                diagram.add_arc(arc)
                    
                    fixed_y_values[var_index] = True

        # Clean diagram
        self.operations.clean_diagram(diagram)

    def HPR_based_compilation(self, instance, diagram, var_order, max_width, discard_method):
        # Stack of nodes
        queue = deque()
        node = diagram.root_node
        while node.id != "sink":
            if node.layer <= instance.Fcols - 1:
                queue.appendleft(node)
            for arc in node.outgoing_arcs:
                if arc.player != None:
                    node = arc.head
        # Compute completion bounds for follower constrs
        initial_completion_bounds = self.initialize_completion_bounds(instance)

        while queue and diagram.width <= max_width:
            node = queue.popleft()
            completion_bounds = initial_completion_bounds.copy()
            # Update completion bounds
            for layer in range(node.layer):
                var_index = var_order["follower"][layer]
                self.operations.update_completions_bounds(instance, completion_bounds, var_index, player="follower")
            # Select player
            player = "follower"
            layer = node.layer
            # Choose player
            if layer >= instance.Fcols:
                player = "leader"
                var_index = var_order[player][layer - instance.Fcols]
            else:
                var_index = var_order[player][layer]

            if player == "follower":
                # Create nodes
                zero_head = self.operations.create_zero_node(layer + 1, parent_node=node)
                one_head = self.operations.create_one_node(instance, layer + 1, var_index, parent_node=node, player="follower")

                # Zero head
                if self.operations.check_completion_bounds(instance, completion_bounds, zero_head) and instance.known_y_values.get(var_index) != 1:
                    zero_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if zero_head.hash_key in diagram.graph_map[zero_head.layer]:
                        found_node = diagram.graph_map[zero_head.layer][zero_head.hash_key]
                        self.operations.update_costs(node=found_node, new_node=zero_head)
                        zero_head = found_node
                        if node.hash_key not in diagram.graph_map[node.layer]:
                            # Create arc
                            arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                    else:
                        diagram.add_node(zero_head)  # TODO: add nodes after reducing next_layer_queue
                        # Create arc
                        arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
                        diagram.add_arc(arc)
                        queue.appendleft(zero_head)
                
                # One head
                if self.operations.check_completion_bounds(instance, completion_bounds, one_head) and instance.known_y_values.get(var_index) != 0:
                    one_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if one_head.hash_key in diagram.graph_map[one_head.layer]:
                        found_node = diagram.graph_map[one_head.layer][one_head.hash_key]
                        self.operations.update_costs(node=found_node, new_node=one_head)
                        one_head = found_node
                        if node.hash_key not in diagram.graph_map[node.layer]:
                            # Create arc
                            arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                    else:
                        diagram.add_node(one_head)  # TODO: add nodes after reducing next_layer_queue
                        # Create arc
                        arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                        diagram.add_arc(arc)
                        queue.appendleft(one_head)

            # Compile compressed leader layer
            else:
                arc = Arc(tail=node, head=diagram.sink_node, value=0, cost=0, var_index=-1, player="leader")
                for i in range(instance.Frows):
                    if instance.interaction[i] == "both":
                        arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
                diagram.add_arc(arc) 

        self.operations.clean_diagram(diagram)
    
    def initialize_completion_bounds(self, instance):
        completion_bounds = np.zeros(instance.Frows) #[0] * instance.Frows
        for i in range(instance.Frows):
            if instance.interaction[i] == "leader":
                completion_bounds[i] = -float("inf")
            else:
                for j in range(instance.Lcols):
                    completion_bounds[i] += min(0, instance.C[i][j])
                    # completion_bounds[i] += HPR_solution["x"][j] * instance.C[i][j] 
                for j in range(instance.Fcols):
                    completion_bounds[i] += min(0, instance.D[i][j])
        
        return completion_bounds
    
    def remove_fixed_state_component(self, instance, fixed_y_values, nodes):
        for i in range(instance.Frows):
            if instance.interaction[i] == "follower":
                if np.all([fixed_y_values[j] for j in range(instance.Fcols) if instance.D[i][j] != 0]):  # All y values have already been set
                    for node in nodes:
                        node.state[i] = -float("inf")