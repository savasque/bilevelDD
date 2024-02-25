from time import time
from collections import deque
import numpy as np

from . import constants

from classes.node import Node
from classes.arc import Arc
from decision_diagram_manager.operations import Operations

from algorithms.utils.solve_HPR import solve as solve_HPR
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
        self.logger.info("Compiling diagram. Compilation method: compressed leader layer - MaxWidth: {} - VarOrder: {}".format(max_width, ordering_heuristic))
        var_order = self.operations.ordering_heuristic(instance, ordering_heuristic)

        diagram.max_width = max_width
        diagram.ordering_heuristic = ordering_heuristic
        diagram.var_order = var_order

        # Queues of nodes
        current_layer_queue = deque()
        next_layer_queue = deque()

        # Create root and sink nodes
        root_node = Node(id="root", layer=0, state=[0] * instance.Frows)
        for i in range(instance.Frows):
            if instance.interaction[i] == "leader":
                root_node.state[i] = None
        diagram.add_node(root_node)
        sink_node = Node(id="sink", layer=instance.Fcols + 1)
        diagram.add_node(sink_node)

        # Create dummy arc or 1-width relaxation
        M, _ = solve_follower_problem(instance, sense="maximize")
        self.logger.debug("Big-M value: {}".format(M))
        dummy_arc = Arc(tail=root_node, head=sink_node, value=0, cost=M, var_index=-1, player=None)
        diagram.add_arc(dummy_arc)


        ##### DD compilation #####

        if not skip_brute_force_compilation:
            ##### Compile y-solutions by binary branching
            self.logger.debug("Compiling new solutions")

            # Compute completion bounds for follower constrs
            completion_bounds = [0] * instance.Frows
            for i in range(instance.Frows):
                if instance.interaction[i] == "leader":
                    completion_bounds[i] = None
                else:
                    for j in range(instance.Lcols):
                        completion_bounds[i] += min(0, instance.C[i][j])
                        # completion_bounds[i] += HPR_optimal_response["x"][j] * instance.C[i][j] 
                    for j in range(instance.Fcols):
                        completion_bounds[i] += min(0, instance.D[i][j])

            # Create new nodes and arcs
            fixed_y_values = {j: False for j in range(instance.Fcols)}
            current_layer_queue.append(root_node)
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

                        # Remove fixed state components
                        for i in range(instance.Frows):
                            if instance.interaction[i] == "follower":
                                if np.all([fixed_y_values[j] for j in range(instance.Fcols) if instance.D[i][j] != 0]):  # All y values have already been set
                                    zero_head.state[i] = None
                                    one_head.state[i] = None

                        # Zero head
                        if self.operations.check_completion_bounds(instance, completion_bounds, zero_head) and instance.known_y_values.get(var_index) != 1:
                            zero_head.id = diagram.node_count + 1
                            # Check if node was already created
                            if zero_head.hash_key in diagram.graph_map:
                                found_node = diagram.graph_map[zero_head.hash_key]
                                self.operations.update_costs(node=found_node, new_node=zero_head)
                                zero_head = found_node
                                if node.hash_key not in diagram.graph_map:
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
                            if one_head.hash_key in diagram.graph_map:
                                found_node = diagram.graph_map[one_head.hash_key]
                                self.operations.update_costs(node=found_node, new_node=one_head)
                                one_head = found_node
                                if node.hash_key not in diagram.graph_map:
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
                        arc = Arc(tail=node, head=sink_node, value=0, cost=0, var_index=-1, player="leader")
                        for i in range(instance.Frows):
                            if instance.interaction[i] == "both":
                                arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
                        diagram.add_arc(arc)

                if diagram.node_count >= constants.MAX_NODE_COUNT:
                    raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

                # Width limit
                if len(next_layer_queue) > max_width:
                    next_layer_queue = self.operations.reduced_queue(instance, discard_method, next_layer_queue, max_width, player)

                # Update queues
                current_layer_queue = next_layer_queue
                next_layer_queue = deque()
        
        # Update in/outgoing arcs
        diagram.update_in_outgoing_arcs()
        # Clean diagram
        self.operations.clean_diagram(diagram)
    

        # ##### Use remaining queues
        # # Update in/outgoing arcs
        # diagram.update_in_outgoing_arcs()
        # # Clean diagram
        # self.operations.clean_diagram(diagram)

        # if diagram.width < max_width:
        #     for last_layer, remaining_queue in remaining_queues.items():
        #         current_layer_queue = deque()
        #         fixed_y_values = {j: False for j in range(instance.Fcols)}

        #         # Add nodes and arcs in remaining queue
        #         while len(current_layer_queue) + len(diagram.layers[last_layer]) < max_width and remaining_queue:
        #             parent_node, node = remaining_queue.popleft()
        #             diagram.add_node(node)
        #             arc = Arc(tail=parent_node, head=node, value=node.type, cost=instance.d[var_index] * node.type, var_index=var_index, player="follower")
        #             diagram.add_arc(arc)
        #             current_layer_queue.append((parent_node, node))
                
        #         # Update completion bounds until current layer
        #         completion_bounds = [0] * instance.Frows
        #         for i in range(instance.Frows):
        #             if instance.interaction[i] == "leader":
        #                 completion_bounds[i] = None
        #             else:
        #                 for j in range(instance.Lcols):
        #                     completion_bounds[i] += min(0, instance.C[i][j])
        #                     # completion_bounds[i] += HPR_optimal_response["x"][j] * instance.C[i][j] 
        #                 for j in range(instance.Fcols):
        #                     completion_bounds[i] += min(0, instance.D[i][j])
        #         for j in range(last_layer + 1):
        #             if j >= instance.Fcols:
        #                 var_index = var_order["leader"][j - instance.Fcols]
        #             else:
        #                 var_index = var_order["follower"][j]
        #             # Update completion bound
        #             self.operations.update_completions_bounds(instance, completion_bounds, var_index, player)
        #             fixed_y_values[var_index] = True

        #         # Create new nodes
        #         for layer in range(last_layer, instance.Fcols + 1):
        #             player = "follower"
        #             if j >= instance.Fcols:
        #                 player = "leader"
        #                 var_index = var_order[player][layer - instance.Fcols]
        #             else:
        #                 var_index = var_order[player][layer]
                    
        #             # Update completion bound
        #             self.operations.update_completions_bounds(instance, completion_bounds, var_index, player)

        #             # Compile follower layers
        #             if player == "follower":
        #                 while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
        #                     _, node = current_layer_queue.popleft()
        #                     zero_head = self.operations.create_zero_node(layer + 1, parent_node=node)
        #                     one_head = self.operations.create_one_node(instance, layer + 1, var_index, parent_node=node, player=player)

        #                     # Remove fixed state components
        #                     for i in range(instance.Frows):
        #                         if instance.interaction[i] == "follower":
        #                             if np.all([fixed_y_values[j] for j in range(instance.Fcols) if instance.D[i][j] != 0]):  # All y values have already been set
        #                                 zero_head.state[i] = None
        #                                 one_head.state[i] = None

        #                     # Zero head
        #                     if self.operations.check_completion_bounds(instance, completion_bounds, zero_head) and instance.known_y_values.get(var_index) != 1:  # TODO: remove this check
        #                         zero_head.id = diagram.node_count + 1
        #                         # Check if node was already created
        #                         if zero_head.hash_key in diagram.graph_map:
        #                             found_node = diagram.nodes[diagram.graph_map[zero_head.hash_key]]
        #                             self.operations.update_costs(node=found_node, new_node=zero_head)
        #                             zero_head = found_node
        #                             if node.hash_key not in diagram.graph_map:
        #                                 # Create arc
        #                                 arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
        #                                 diagram.add_arc(arc)
        #                         else:
        #                             diagram.add_node(zero_head)  # TODO: add nodes after reducing next_layer_queue
        #                             # Create arc
        #                             arc = Arc(tail=node, head=zero_head, value=0, cost=0, var_index=var_index, player="follower")
        #                             diagram.add_arc(arc)
        #                             next_layer_queue.append(zero_head)
                            
        #                     # One head
        #                     if self.operations.check_completion_bounds(instance, completion_bounds, one_head) and instance.known_y_values.get(var_index) != 0:
        #                         one_head.id = diagram.node_count + 1
        #                         # Check if node was already created
        #                         if one_head.hash_key in diagram.graph_map:
        #                             found_node = diagram.nodes[diagram.graph_map[one_head.hash_key]]
        #                             self.operations.update_costs(node=found_node, new_node=one_head)
        #                             one_head = found_node
        #                             if node.hash_key not in diagram.graph_map:
        #                                 # Create arc
        #                                 arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
        #                                 diagram.add_arc(arc)
        #                         else:
        #                             diagram.add_node(one_head)  # TODO: add nodes after reducing next_layer_queue
        #                             # Create arc
        #                             arc = Arc(tail=node, head=one_head, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
        #                             diagram.add_arc(arc)
        #                             next_layer_queue.append(one_head)

        #                 # next_layer_queue = deque(next_layer_hash.values())
        #                 fixed_y_values[var_index] = True

        #             # Compile compressed leader layer
        #             else:
        #                 while len(current_layer_queue):
        #                     node = current_layer_queue.popleft()
        #                     arc = Arc(tail=node, head=sink_node, value=0, cost=0, var_index=-1, player="leader")
        #                     for i in range(instance.Frows):
        #                         if instance.interaction[i] == "both":
        #                             arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
        #                     diagram.add_arc(arc)

        #         # Update in/outgoing arcs
        #         diagram.update_in_outgoing_arcs()
        #         # Clean diagram
        #         diagram = self.operations.clean_diagram(diagram)
        #         a = None
                

        ##### Compile y-solutions within Y
        self.logger.debug("Compiling y-solutions ({}) in set Y".format(len(Y)))
        for idx, y in enumerate(Y):
            if diagram.width <= max_width:
                self.logger.debug("Solution {}/{}".format(idx + 1, len(Y)))
                fixed_y_values = {j: False for j in range(instance.Fcols)}
                child_node = diagram.root_node

                for layer in range(instance.Fcols):
                    var_index = var_order["follower"][layer]
                
                    # Create nodes
                    if diagram.node_count < constants.MAX_NODE_COUNT:
                        node = child_node
                        if y[var_index] == 0:
                            child_node = self.operations.create_zero_node(layer + 1, node)
                        else:
                            child_node = self.operations.create_one_node(instance, layer + 1, var_index, node, player="follower")

                        # Remove fixed state components
                        for i in range(instance.Frows):
                            if instance.interaction[i] == "follower":
                                if np.all([fixed_y_values[j] for j in range(instance.Fcols) if instance.D[i][j] != 0]):  # All y values have already been set
                                    child_node.state[i] = None

                        # Zero head
                        if y[var_index] == 0:
                            child_node.id = diagram.node_count + 1
                            # Check if node was already created
                            if child_node.hash_key in diagram.graph_map:
                                found_node = diagram.graph_map[child_node.hash_key]
                                self.operations.update_costs(node=found_node, new_node=child_node)
                                child_node = found_node
                                if node.hash_key not in diagram.graph_map:
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
                                    arc = Arc(tail=child_node, head=sink_node, value=0, cost=0, var_index=-1, player="leader")
                                    for i in range(instance.Frows):
                                        if instance.interaction[i] == "both":
                                            arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
                                    diagram.add_arc(arc)
                        
                        # One head
                        else:
                            child_node.id = diagram.node_count + 1
                            # Check if node was already created
                            if child_node.hash_key in diagram.graph_map:
                                found_node = diagram.graph_map[child_node.hash_key]
                                self.operations.update_costs(node=found_node, new_node=child_node)
                                child_node = found_node
                                if node.hash_key not in diagram.graph_map:
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
                                    arc = Arc(tail=child_node, head=sink_node, value=0, cost=0, var_index=-1, player="leader")
                                    for i in range(instance.Frows):
                                        if instance.interaction[i] == "both":
                                            arc.block_values[i] = instance.b[i] - child_node.state[i] + 1 
                                    diagram.add_arc(arc)
                        
                        fixed_y_values[var_index] = True
            else:
                self.logger.debug("Max width reached")

        # Update in/outgoing arcs
        diagram.update_in_outgoing_arcs()
        # Clean diagram
        self.operations.clean_diagram(diagram)

        diagram.compilation_method = "follower_then_compressed_leader"
        diagram.compilation_runtime = time() - t0
        self.logger.info("Diagram succesfully compiled. Time elapsed: {} s - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, diagram.node_count + 2, diagram.arc_count, diagram.width
        ))
        self.logger.info("Finishing compilation process. Time elapsed: {} s".format(diagram.compilation_runtime))

        return diagram