import logzero
from collections import deque
from time import time

from classes.node import Node
from classes.arc import Arc
from classes.decision_diagram import DecisionDiagram

from algorithms.utils.solve_HPR import run as solve_HPR


class DecisionDiagramManager:
    def __init__(self):
        self.logger = logzero.logger

    def compile_diagram(self, diagram, instance, compilation, max_width):
        '''
            This method compiles a DD, starting with the follower and continuing with the leader.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int)
            Returns: diagram (class DecisionDiagram)
        '''

        t0 = time()
        self.logger.info("Compiling diagram. Compilation method: {}".format(compilation))
        var_order = self.ordering_heuristic(instance)
        n = instance.Lcols + instance.Fcols
        player = "follower"

        # Queues of nodes
        current_layer_queue = deque()
        next_layer_queue = deque()

        # Create root and sink nodes
        root_node = Node(id="root", layer=0, state=[0] * instance.Frows)
        diagram.add_node(root_node)
        current_layer_queue.append(root_node)
        sink_node = Node(id="sink", layer=n)
        diagram.add_node(sink_node)

        # Create dummy arc or 1-width relaxation
        M = solve_HPR(instance, obj="follower", sense="max") - solve_HPR(instance, obj="follower", sense="min")
        self.logger.debug("Big-M value: {}".format(M))
        dummy_arc = Arc(tail=root_node.id, head=sink_node.id, value=0, cost=M, var_index=-1, player=None)
        diagram.add_arc(dummy_arc)

        # Compute completion bounds for follower constrs
        completion_bounds = [0] * instance.Frows
        for i in range(instance.Frows):
            for j in range(instance.Fcols):
                completion_bounds[i] += min(0, instance.C[i][j]) + min(0, instance.D[i][j])
        self.logger.debug("Completion bounds for follower layers: {}".format(completion_bounds)) 

        ## Build follower layers ##   
        # Create new nodes and arcs
        for layer in range(n):
            if layer >= instance.Fcols:
                player = "leader"
                var_index = var_order[player][layer - instance.Fcols]
            else:
                var_index = var_order[player][layer]
            self.logger.debug("{} layer: {} - Variable index: {} - Queue size: {}".format(player, layer, var_index, len(current_layer_queue)))
            # Update completion bound
            self.update_completions_bounds(instance, completion_bounds, var_index, player)
            while len(current_layer_queue) and diagram.node_count < 1e6:
                node = current_layer_queue.popleft()
                zero_head = self.create_zero_node(layer + 1, node)
                one_head = self.create_one_node(instance, layer + 1, var_index, node, player=player)

                # Zero head
                if self.check_completion_bounds(instance, completion_bounds, zero_head):
                    zero_head.id = diagram.node_count
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
                    one_head.id = diagram.node_count
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

            # Width limit
            if compilation == "restricted" and len(next_layer_queue) > max_width:
                next_layer_queue = self.reduced_queue(diagram, next_layer_queue, max_width, player=player)
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

        self.logger.info("Diagram succesfully compiled. Time elapse: {} sec".format(time() - t0))

        return clean_diagram

    def ordering_heuristic(self, instance):
        '''
            This method retrieves a variable ordering.

            Args: instance (class Instance)
            Returns: dict of sorted indices (dict)
        '''

        order = {"leader": list(), "follower": list()}

        # Sum coeffs of the left hand side
        for j in range(instance.Fcols):
            coeffs_sum = sum(instance.B[i][j] + instance.D[i][j] for i in range(instance.Frows))
            order["follower"].append((j, coeffs_sum))
        for j in range(instance.Lcols):
            coeffs_sum = sum(instance.A[i][j] + instance.C[i][j] for i in range(instance.Frows))
            order["leader"].append((j, coeffs_sum))
        for key in order:  # Sort variables in descending order
            order[key].sort(key=lambda x: x[1], reverse=True)
            order[key] = [i[0] for i in order[key]]

        self.logger.debug("Variable ordering: {}".format(order))

        return order
    
    def create_zero_node(self, layer, parent_node):
        node = Node(id=None, layer=layer, state=parent_node.state)
        node.leader_cost = parent_node.leader_cost
        node.follower_cost = parent_node.follower_cost

        return node
    
    def create_one_node(self, instance, layer, var_index, parent_node, player):
        node = Node(id=None, layer=layer, state=list(parent_node.state))
        if player == "follower":
            for i in range(instance.Frows):
                node.state[i] += instance.D[i][var_index]
            node.leader_cost = parent_node.leader_cost + instance.c_follower[var_index]
            node.follower_cost = parent_node.follower_cost + instance.d[var_index]
        else:
            for i in range(instance.Frows):
                node.state[i] += instance.C[i][var_index]
            node.leader_cost = parent_node.leader_cost + instance.c_leader[var_index]
            node.follower_cost = parent_node.follower_cost

        return node
    
    def check_completion_bounds(self, instance, completion_bounds, node):
        for i in range(instance.Frows):
            if node.state[i] + completion_bounds[i] > instance.b[i]:
                return False
        
        return True

    def completion_bounds_sanity_check(self, instance, node):
        '''
            This method checks if any infeasible r-t path was compiled.

            Args: instance (class Instance), node (class Node).
            Returns: None
        '''
        
        for i in range(instance.Frows):
            if node.state[i] > instance.b[i]:
                raise ValueError("Infeasible path. NodeID: {} - State: {}".format(node.id, node.state))
            
    def update_completions_bounds(self, instance, completion_bounds, var_index, player):
        for i in range(instance.Frows):
            if player == "follower":
                completion_bounds[i] -= min(0, instance.D[i][var_index])
            else:
                completion_bounds[i] -= min(0, instance.C[i][var_index])

    def update_costs(self, node, new_node):
        node.leader_cost = min(node.leader_cost, new_node.leader_cost)
        node.follower_cost = min(node.follower_cost, new_node.follower_cost)

    def reduced_queue(self, diagram, queue, max_width, player):
        if player == "follower":
            queue = sorted(queue, key=lambda x: int(0.9 * x.follower_cost + 0.1 * x.leader_cost))  # Sort nodes in increasing order # TODO: remove int
        else:
            queue = sorted(queue, key=lambda x: x.leader_cost)  # Sort nodes in increasing order
        
        return deque(queue[:max_width])

    def clean_diagram(self, diagram):
        clean_diagram = DecisionDiagram()
        # Add root and last added node
        root_node = diagram.nodes["root"]
        clean_diagram.add_node(root_node)
        sink_node = diagram.nodes["sink"]
        clean_diagram.add_node(sink_node)

        self.bottom_up_recursion(diagram, clean_diagram, sink_node)

        # Update arc pointers
        for arc in clean_diagram.arcs:
            arc.tail = clean_diagram.graph_map[diagram.nodes[arc.tail].hash_key]
            arc.head = clean_diagram.graph_map[diagram.nodes[arc.head].hash_key]
        
        # Update in and outgoing arcs
        clean_diagram.remove_in_outgoing_arcs()
        for arc in clean_diagram.arcs:
            clean_diagram.nodes[arc.tail].outgoing_arcs.append(arc)
            clean_diagram.nodes[arc.head].incoming_arcs.append(arc)

        self.logger.debug("Diagram succesfully compiled: Node count: {} - Arc count: {}".format(clean_diagram.node_count, clean_diagram.arc_count))

        return clean_diagram

    def bottom_up_recursion(self, diagram, clean_diagram, node):
        if node.hash_key not in clean_diagram.graph_map:
            clean_diagram.add_node(node)
        for arc in node.incoming_arcs:
            tail_node = diagram.nodes[arc.tail]
            if tail_node.hash_key in diagram.graph_map:
                if tail_node.hash_key not in clean_diagram.graph_map:
                    self.bottom_up_recursion(diagram, clean_diagram, tail_node)
                clean_diagram.add_arc(arc)