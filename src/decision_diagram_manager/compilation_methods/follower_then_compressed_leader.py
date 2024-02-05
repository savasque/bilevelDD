from time import time
from collections import deque
import numpy as np

from . import constants

from classes.node import Node
from classes.arc import Arc
from decision_diagram_manager.operations import Operations

from algorithms.utils.solve_HPR import solve as solve_HPR

class FollowerThenCompressedLeaderCompiler:
    def __init__(self, logger):
        self.logger = logger
        self.operations = Operations(logger)

    def compile(self, diagram, instance, max_width, ordering_heuristic, Y):
        """
            This method compiles a DD, starting with the follower and continuing with a single compressed leader layer.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int), ordering_heuristic (str), [optional] Y (list)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation method: compressed leader - MaxWidth: {}".format(max_width))
        var_order = self.operations.ordering_heuristic(instance, ordering_heuristic)
        self.logger.debug("Variable ordering: {}".format(var_order))
        player = "follower"

        diagram.max_width = max_width
        diagram.ordering_heuristic = ordering_heuristic
        diagram.var_order = var_order

        # Queues of nodes
        current_layer_queue = deque()
        inbetween_layer_queue = dict()
        next_layer_queue = deque()

        # Create root and sink nodes
        root_node = Node(id="root", layer=0, state=[0] * instance.Frows)
        diagram.add_node(root_node)
        sink_node = Node(id="sink", layer=instance.Fcols + 1)
        diagram.add_node(sink_node)

        # Create dummy arc or 1-width relaxation
        M = solve_HPR(instance, obj="follower", sense="max")[0]
        self.logger.debug("Big-M value: {}".format(M))
        dummy_arc = Arc(tail=root_node.id, head=sink_node.id, value=0, cost=M, var_index=-1, player=None)
        diagram.add_arc(dummy_arc)

        ##### Build layers #####
        ## First, compile y-solutions within Y
        self.logger.debug("Compiling y-solutions ({}) in set Y".format(len(Y)))
        for idx, y in enumerate(Y):
            current_layer_queue.append(root_node)
            for layer in range(instance.Fcols):
                next_layer_hash = dict()
                var_index = var_order["follower"][layer]
                self.logger.debug("Follower ({}/{}) layer for given solution: {} - Variable index: {} - Queue size: {}".format(idx + 1, len(Y), layer, var_index, len(current_layer_queue)))
            
                # Create nodes
                while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
                    node = current_layer_queue.popleft()
                    if y[var_index] == 0:
                        child_node = self.operations.create_zero_node(layer + 1, node)
                    else:
                        child_node = self.operations.create_one_node(instance, layer + 1, var_index, node, player="follower")

                    # Zero head
                    if y[var_index] == 0:
                        child_node.id = diagram.node_count + 1
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[child_node.hash_key]]
                            self.operations.update_costs(node=found_node, new_node=child_node)
                            next_layer_queue.append(found_node)
                            child_node = found_node
                            if node.hash_key not in diagram.graph_map:
                                # Create arc
                                arc = Arc(tail=node.id, head=child_node.id, value=0, cost=0, var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(child_node)
                            next_layer_queue.append(child_node)
                            # Create arc
                            arc = Arc(tail=node.id, head=child_node.id, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                        next_layer_hash[zero_head.hash_key] = zero_head
                    
                    # One head
                    else:
                        child_node.id = diagram.node_count + 1
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[child_node.hash_key]]
                            self.operations.update_costs(node=found_node, new_node=child_node)
                            next_layer_queue.append(found_node)
                            child_node = found_node
                            if node.hash_key not in diagram.graph_map:
                                # Create arc
                                arc = Arc(tail=node.id, head=child_node.id, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(child_node)
                            next_layer_queue.append(child_node)
                            # Create arc
                            arc = Arc(tail=node.id, head=child_node.id, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                        next_layer_hash[child_node.hash_key] = child_node

                # Width limit
                if len(next_layer_queue) > max_width:
                    next_layer_queue = self.reduced_queue(next_layer_queue, max_width, player="follower")
                if layer == instance.Fcols - 1 and next_layer_queue:
                    inbetween_layer_queue[next_layer_queue[0].hash_key] = next_layer_queue[0]
                    current_layer_queue = deque()
                else:
                    current_layer_queue = next_layer_queue
                next_layer_queue = deque()

        ## Then, start again, compile more y-solutions, and compile the corresponding x-compressed layers
        self.logger.debug("Compiling new solutions")

        # Compute completion bounds for follower constrs
        completion_bounds = [0] * instance.Frows
        for i in range(instance.Frows):
            for j in range(instance.Lcols):
                completion_bounds[i] += min(0, instance.C[i][j])
            for j in range(instance.Fcols):
                completion_bounds[i] += min(0, instance.D[i][j])
        # self.logger.debug("Completion bounds for follower constrs: {}".format(completion_bounds)) 

        # Create new nodes and arcs
        current_layer_queue.append(root_node)
        for layer in range(instance.Fcols + 1):
            next_layer_hash = dict()  # To avoid duplicates in next_layer_queue

            # Choose player
            if layer >= instance.Fcols:
                player = "leader"
                var_index = var_order[player][layer - instance.Fcols]
            else:
                var_index = var_order[player][layer]
            self.logger.debug("{} layer: {} - Variable index: {} - Queue size: {}".format(player.capitalize(), layer, var_index, len(current_layer_queue)))

            # Update completion bound
            self.operations.update_completions_bounds(instance, completion_bounds, var_index, player)

            # Compile follower layers
            if player == "follower":
                while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
                    node = current_layer_queue.popleft()
                    zero_head = self.operations.create_zero_node(layer + 1, node)
                    one_head = self.operations.create_one_node(instance, layer + 1, var_index, node, player=player)

                    # Zero head
                    if self.operations.check_completion_bounds(instance, completion_bounds, zero_head):
                        zero_head.id = diagram.node_count + 1
                        # Check if node was already created
                        if zero_head.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[zero_head.hash_key]]
                            self.operations.update_costs(node=found_node, new_node=zero_head)
                            zero_head = found_node
                            if node.hash_key not in diagram.graph_map:
                                # Create arc
                                arc = Arc(tail=node.id, head=zero_head.id, value=0, cost=0, var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(zero_head)
                            # Create arc
                            arc = Arc(tail=node.id, head=zero_head.id, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                        next_layer_hash[zero_head.hash_key] = zero_head
                    
                    # One head
                    if self.operations.check_completion_bounds(instance, completion_bounds, one_head):
                        one_head.id = diagram.node_count + 1
                        # Check if node was already created
                        if one_head.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[one_head.hash_key]]
                            self.operations.update_costs(node=found_node, new_node=one_head)
                            one_head = found_node
                            if node.hash_key not in diagram.graph_map:
                                # Create arc
                                arc = Arc(tail=node.id, head=one_head.id, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(one_head)
                            # Create arc
                            arc = Arc(tail=node.id, head=one_head.id, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                        next_layer_hash[one_head.hash_key] = one_head

                next_layer_queue = deque(next_layer_hash.values())

            # Compile compressed leader layer
            else:
                while len(current_layer_queue):
                    node = current_layer_queue.popleft()
                    arc = Arc(tail=node.id, head=sink_node.id, value=0, cost=0, var_index=-1, player="leader")
                    for i in range(instance.Frows):
                        if np.any(instance.C[i]) and np.any(instance.D[i]):
                            arc.block_values[i] = instance.b[i] - node.state[i] + 1  # TODO: check the +1
                    diagram.add_arc(arc)

            if diagram.node_count >= constants.MAX_NODE_COUNT:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if len(next_layer_queue) > max_width:
                next_layer_queue = self.operations.reduced_queue(next_layer_queue, max_width, player=player)

            # Update queues
            current_layer_queue = next_layer_queue
            next_layer_queue = deque()

        # Update in/outgoing arcs
        diagram.update_in_outgoing_arcs()

        # Clean diagram
        clean_diagram = self.operations.clean_diagram(diagram)

        clean_diagram.compilation_runtime = time() - t0

        self.logger.info("Diagram succesfully compiled. Time elapsed: {} s - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, clean_diagram.node_count + 2, clean_diagram.arc_count, clean_diagram.width
        ))

        return clean_diagram