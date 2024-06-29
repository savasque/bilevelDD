import logzero
from collections import deque
from time import time
import numpy as np

from .compilation_methods import constants

from classes.node import Node
from classes.arc import Arc
from classes.decision_diagram import DecisionDiagram

from algorithms.utils.solve_HPR import solve as solve_HPR

from .compilation_methods.follower_then_full_leader import FollowerThenLeaderCompiler
from .compilation_methods.follower_then_leader import FollowerThenCompressedLeaderCompiler

class DecisionDiagramManager:
    def __init__(self):
        self.logger = logzero.logger
        self.compilation_methods = {
            "branching": self.compile_diagram_follower_then_compressed_leader,
            "iterative": self.compile_diagram_iteratively
        }

    def compile_diagram(self, diagram, instance, ordering_heuristic, method, max_width, discard_method):
        diagram = self.compilation_methods[method](diagram, instance, max_width, ordering_heuristic, discard_method)
        
        return diagram

    def compile_diagram_follower_then_leader(self, diagram, instance, max_width, ordering_heuristic, HPR_optimal_response, Y):
        compiler = FollowerThenLeaderCompiler(self.logger)
        compiled_diagram = compiler.compile(diagram, instance, max_width, ordering_heuristic, HPR_optimal_response, Y)
        
        return compiled_diagram

    def compile_diagram_leader_then_follower(self, diagram, instance, max_width, ordering_heuristic, Y):
        """
            This method compiles a DD, starting with the leader and continuing with the follower.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation method: leader-follower")
        var_order = self.ordering_heuristic(instance, ordering_heuristic)
        self.logger.debug("Variable ordering: {}".format(var_order))
        n = instance.Lcols + instance.Fcols
        player = "leader"

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
        self.logger.debug("Completion bounds for follower constrs: {}".format(completion_bounds)) 

        ## Build layers 
        # Create new nodes and arcs
        for layer in range(n):
            # Choose player
            if layer >= instance.Lcols:
                player = "follower"
                var_index = var_order[player][layer - instance.Lcols]
            else:
                var_index = var_order[player][layer]
            self.logger.debug("{} layer: {} - Variable index: {} - Queue size: {}".format(player, layer, var_index, len(current_layer_queue)))

            # Update completion bound
            self.update_completions_bounds(instance, completion_bounds, var_index, player)
            while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
                node = current_layer_queue.popleft()
                zero_head = self.create_zero_node(layer + 1, node)
                one_head = self.create_one_node(instance, layer + 1, var_index, node, player=player)

                # Zero head
                if self.check_completion_bounds(instance, completion_bounds, zero_head):
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
                if self.check_completion_bounds(instance, completion_bounds, one_head):
                    one_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if one_head.hash_key in diagram.graph_map:
                        found_node = diagram.nodes[diagram.graph_map[one_head.hash_key]]
                        if player == "follower":
                            self.update_costs(node=found_node, new_node=one_head)
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
                next_layer_queue = self.reduced_queue(next_layer_queue, max_width, player=player)
            current_layer_queue = next_layer_queue
            next_layer_queue = deque()

        # Add artificial zero arcs to the sink node
        while len(current_layer_queue):
            node = current_layer_queue.popleft()
            self.completion_bounds_sanity_check(instance, node)
            arc = Arc(tail=node.id, head=sink_node.id, value=0, cost=0, var_index=-1, player=None)
            diagram.add_arc(arc)

        # Update in/outgoing arcs
        diagram.update_in_outgoing_arcs()

        # Clean diagram
        clean_diagram = self.clean_diagram(diagram)
        clean_diagram.initial_width = int(diagram.width)

        clean_diagram.compilation_runtime = time() - t0

        self.logger.info("Diagram succesfully compiled. Time elapsed: {} - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, clean_diagram.node_count + 2, clean_diagram.arc_count, clean_diagram.width
        ))

        return clean_diagram

    def compile_diagram_iteratively(self, diagram, instance, max_width, ordering_heuristic, Y):
        """
            This method compiles a DD, starting with the follower and continuing with the leader. 
            It only compiles y's belonging to set Y.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), Y (list), max_width (int)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation method: iterative/collect_Y - MaxWidth: {}".format(max_width))
        var_order = self.ordering_heuristic(instance, ordering_heuristic)
        self.logger.debug("Variable ordering: {}".format(var_order))
        n = instance.Lcols + instance.Fcols

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
        self.logger.debug("Completion bounds for follower constrs: {}".format(completion_bounds)) 

        ## Build follower layers ##   
        # Create new nodes and arcs
        for idx, y in enumerate(Y):
            current_layer_queue.append(root_node)
            for layer in range(instance.Fcols):
                completion_bounds_temp = list(completion_bounds)
                var_index = var_order["follower"][layer]
                self.logger.debug("follower ({}/{}) layer: {} - Variable index: {} - Queue size: {}".format(idx + 1, len(Y), layer, var_index, len(current_layer_queue)))
                
                # Update completion bound
                self.update_completions_bounds(instance, completion_bounds_temp, var_index, player="follower")
                while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
                    node = current_layer_queue.popleft()
                    if y[var_index] == 0:
                        child_node = self.create_zero_node(layer + 1, node)
                    else:
                        child_node = self.create_one_node(instance, layer + 1, var_index, node, player="follower")

                    # Zero head
                    if self.check_completion_bounds(instance, completion_bounds, child_node) and y[var_index] == 0:
                        child_node.id = diagram.node_count + 1
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[child_node.hash_key]]
                            self.update_costs(node=found_node, new_node=child_node)
                            next_layer_queue.append(found_node)
                            if node.hash_key not in diagram.graph_map:
                                # Create arc
                                arc = Arc(tail=node.id, head=found_node.id, value=0, cost=0, var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(child_node)
                            next_layer_queue.append(child_node)
                            # Create arc
                            arc = Arc(tail=node.id, head=child_node.id, value=0, cost=0, var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                    
                    # One head
                    if self.check_completion_bounds(instance, completion_bounds, child_node) and y[var_index] == 1:
                        child_node.id = diagram.node_count + 1
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[child_node.hash_key]]
                            self.update_costs(node=found_node, new_node=child_node)
                            next_layer_queue.append(found_node)
                            if node.hash_key not in diagram.graph_map:
                                # Create arc
                                arc = Arc(tail=node.id, head=found_node.id, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                                diagram.add_arc(arc)
                        else:
                            diagram.add_node(child_node)
                            next_layer_queue.append(child_node)
                            # Create arc
                            arc = Arc(tail=node.id, head=child_node.id, value=1, cost=instance.d[var_index], var_index=var_index, player="follower")
                            diagram.add_arc(arc)
                        

                # Width limit
                if len(next_layer_queue) > max_width:
                    next_layer_queue = self.reduced_queue(next_layer_queue, max_width, player="follower")
                if layer == instance.Fcols - 1 and next_layer_queue:
                    inbetween_layer_queue[next_layer_queue[0].hash_key] = next_layer_queue[0]
                    current_layer_queue = deque()
                else:
                    current_layer_queue = next_layer_queue
                next_layer_queue = deque()
        
        ## Build leader layers ##   
        # Create new nodes and arcs
        for layer in range(instance.Fcols):
            var_index = var_order["follower"][layer]
            self.update_completions_bounds(instance, completion_bounds, var_index, player="follower")
        current_layer_queue = deque(inbetween_layer_queue.values())
        for layer in range(instance.Fcols, instance.Fcols + instance.Lcols):
            var_index = var_order["leader"][layer - instance.Fcols]
            self.logger.debug("leader layer: {} - Variable index: {} - Queue size: {}".format(layer, var_index, len(current_layer_queue)))
            # Update completion bound
            self.update_completions_bounds(instance, completion_bounds, var_index, player="leader")
            while len(current_layer_queue) and diagram.node_count < constants.MAX_NODE_COUNT:
                node = current_layer_queue.popleft()
                zero_head = self.create_zero_node(layer + 1, node)
                one_head = self.create_one_node(instance, layer + 1, var_index, node, player="leader")

                # Zero head
                if self.check_completion_bounds(instance, completion_bounds, zero_head):
                    zero_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if zero_head.hash_key in diagram.graph_map:
                        found_node = diagram.nodes[diagram.graph_map[zero_head.hash_key]]
                        zero_head = found_node
                    else:
                        diagram.add_node(zero_head)
                        next_layer_queue.append(zero_head)
                    # Create arc
                    arc = Arc(tail=node.id, head=zero_head.id, value=0, cost=0, var_index=var_index, player="leader")
                    diagram.add_arc(arc)
                
                # One head
                if self.check_completion_bounds(instance, completion_bounds, one_head):
                    one_head.id = diagram.node_count + 1
                    # Check if node was already created
                    if one_head.hash_key in diagram.graph_map:
                        found_node = diagram.nodes[diagram.graph_map[one_head.hash_key]]
                        one_head = found_node
                    else:
                        diagram.add_node(one_head)
                        next_layer_queue.append(one_head)
                    # Create arc
                    arc = Arc(tail=node.id, head=one_head.id, value=1, cost=0, var_index=var_index, player="leader")
                    diagram.add_arc(arc)

            if diagram.node_count >= constants.MAX_NODE_COUNT:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if len(next_layer_queue) > max_width:
                next_layer_queue = self.reduced_queue(next_layer_queue, max_width, player="leader")
            current_layer_queue = next_layer_queue
            next_layer_queue = deque()

        # Add artificial zero arcs to the sink node
        while len(current_layer_queue):
            node = current_layer_queue.popleft()
            self.completion_bounds_sanity_check(instance, node)
            arc = Arc(tail=node.id, head=sink_node.id, value=0, cost=0, var_index=-1, player=None)
            diagram.add_arc(arc)

        # Update in/outgoing arcs
        diagram.update_in_outgoing_arcs()

        # Clean diagram
        clean_diagram = self.clean_diagram(diagram)

        clean_diagram.compilation_runtime = time() - t0

        self.logger.info("Diagram succesfully compiled. Time elapsed: {} - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, clean_diagram.node_count + 2, clean_diagram.arc_count, clean_diagram.width
        ))

        return clean_diagram

    def compile_diagram_follower_then_compressed_leader(self, diagram, instance, max_width, ordering_heuristic, discard_method):
        compiler = FollowerThenCompressedLeaderCompiler(self.logger)
        compiled_diagram = compiler.compile(diagram, instance, max_width, ordering_heuristic, discard_method)
        
        return compiled_diagram