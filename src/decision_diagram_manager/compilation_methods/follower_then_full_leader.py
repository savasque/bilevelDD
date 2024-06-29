from time import time
from collections import deque
import numpy as np

from . import constants

from classes.node import Node
from classes.arc import Arc
from decision_diagram_manager.operations import Operations

from algorithms.utils.solve_HPR import solve as solve_HPR

class FollowerThenLeaderCompiler:
    def __init__(self, logger):
        self.logger = logger
        self.operations = Operations(logger)

    def compile(self, diagram, instance, max_width, ordering_heuristic, HPR_optimal_response, Y):
        """
            This method compiles a DD, starting with the follower and continuing with the leader.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int), ordering_heuristic (str), [optional] Y (list)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation method: follower-then-leader - MaxWidth: {}".format(max_width))
        self.logger.debug("Variable ordering heuristic: {}".format(ordering_heuristic or "lexicographic ordering"))
        var_order = self.operations.ordering_heuristic(instance, ordering_heuristic)
        # logger.debug("Variable ordering: {}".format(var_order))
        n = instance.Lcols + instance.Fcols
        player = "follower"

        diagram.max_width = max_width
        diagram.ordering_heuristic = ordering_heuristic
        diagram.var_order = var_order

        # Queues of nodes
        current_layer_queue = deque()
        next_layer_queue = deque()

        # Create root and sink nodes
        root_node = Node(id="root", layer=0, state=[0] * instance.Frows)
        diagram.add_node(root_node)
        current_layer_queue.append(root_node)
        sink_node = Node(id="sink", layer=n + 1)
        diagram.add_node(sink_node)

        # Create dummy arc or 1-width relaxation
        M = solve_HPR(instance, obj="follower", sense="max")[0]
        self.logger.debug("Big-M value: {}".format(M))
        dummy_arc = Arc(tail=root_node.id, head=sink_node.id, value=0, cost=M, var_index=-1, player=None)
        diagram.add_arc(dummy_arc)

        # Compute completion bounds for follower constrs
        completion_bounds = [0] * instance.Frows
        for i in range(instance.Frows):
            for j in range(instance.Lcols):
                completion_bounds[i] += min(0, instance.C[i][j])
            for j in range(instance.Fcols):
                completion_bounds[i] += min(0, instance.D[i][j])

        ## Build layers ##   
        # Create new nodes and arcs
        for layer in range(n):
            # Choose player
            if layer >= instance.Fcols:
                player = "leader"
                var_index = var_order[player][layer - instance.Fcols]
            else:
                var_index = var_order[player][layer]
            self.logger.debug("{} layer: {} - Variable index: {} - Queue size: {}".format(player, layer, var_index, len(current_layer_queue)))

            # Update completion bound
            self.operations.update_completions_bounds(instance, completion_bounds, var_index, player)

            # Create nodes
            while len(current_layer_queue) and diagram.node_count < 1e8:
                node = current_layer_queue.popleft()
                zero_head = self.operations.create_zero_node(layer + 1, node)
                one_head = self.operations.create_one_node(instance, layer + 1, var_index, node, player=player)

                # Zero head
                if self.operations.check_completion_bounds(instance, completion_bounds, zero_head):
                    zero_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if zero_head.hash_key in diagram.graph_map:
                        found_node = diagram.nodes[diagram.graph_map[zero_head.hash_key]]
                        if player == "follower":
                            self.update_costs(node=found_node, new_node=zero_head)
                        zero_head = found_node
                    else:
                        diagram.add_node(zero_head)
                        next_layer_queue.append(zero_head)
                    # Create arc
                    arc = Arc(tail=node.id, head=zero_head.id, value=0, cost=0, var_index=var_index, player=player)
                    diagram.add_arc(arc)
                
                # One head
                if self.operations.check_completion_bounds(instance, completion_bounds, one_head):
                    one_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if one_head.hash_key in diagram.graph_map:
                        found_node = diagram.nodes[diagram.graph_map[one_head.hash_key]]
                        if player == "follower":
                            self.operations.update_costs(node=found_node, new_node=one_head)
                        one_head = found_node
                    else:
                        diagram.add_node(one_head)
                        next_layer_queue.append(one_head)
                    # Create arc
                    arc = Arc(tail=node.id, head=one_head.id, value=1, cost=instance.d[var_index] if player=="follower" else 0, var_index=var_index, player=player)
                    diagram.add_arc(arc)

            if diagram.node_count >= constants.MAX_NODE_COUNT:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if len(next_layer_queue) > max_width:
                next_layer_queue = self.operations.reduced_queue(next_layer_queue, max_width, player=player)
            current_layer_queue = next_layer_queue
            next_layer_queue = deque()

        # Add artificial zero arcs to the sink node
        while len(current_layer_queue):
            node = current_layer_queue.popleft()
            self.operations.completion_bounds_sanity_check(instance, node)
            arc = Arc(tail=node.id, head=sink_node.id, value=0, cost=0, var_index=-1, player=None)
            diagram.add_arc(arc)

        # Update in/outgoing arcs
        diagram.update_in_outgoing_arcs()

        # Clean diagram
        clean_diagram = self.operations.clean_diagram(diagram)
        clean_diagram.initial_width = int(clean_diagram.width)
        clean_diagram.compilation_method = "follower_then_leader"

        self.logger.info("Diagram succesfully compiled. Time elapsed: {} s - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, clean_diagram.node_count + 2, clean_diagram.arc_count, clean_diagram.width
        ))

        # Reduce diagram
        self.operations.reduce_diagram(clean_diagram)

        clean_diagram.compilation_runtime = time() - t0
        self.logger.info("Finishing compilation process. Time elapsed: {} s".format(clean_diagram.compilation_runtime))

        return clean_diagram