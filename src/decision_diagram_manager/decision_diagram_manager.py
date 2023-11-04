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
        self.compilation_methods = {
            "follower_leader": self.compile_diagram_FL,
            "leader_follower": self.compile_diagram_LF,
            "iterative": self.compile_diagram_iteratively
        }

    def compile_diagram(self, diagram, instance, compilation, compilation_method, max_width, ordering_heuristic, Y=None):
        if Y and compilation_method in ["collect_Y", "iterative"]:
            diagram = self.compilation_methods["iterative"](diagram, instance, compilation, max_width, ordering_heuristic, Y)
        else:
            diagram = self.compilation_methods[compilation_method](diagram, instance, compilation, max_width, ordering_heuristic)
        
        diagram.initial_width = int(diagram.width)

        self.reduce_diagram(diagram)
        
        return diagram

    def compile_diagram_FL(self, diagram, instance, compilation, max_width, ordering_heuristic):
        """
            This method compiles a DD, starting with the follower and continuing with the leader.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation type: {} - Compilation method: follower-leader - MaxWidth: {}".format(compilation, max_width))
        var_order = self.ordering_heuristic(instance, ordering_heuristic)
        self.logger.debug("Variable ordering: {}".format(var_order))
        n = instance.Lcols + instance.Fcols
        player = "follower"

        diagram.max_width = max_width
        diagram.compilation = compilation
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
            for j in range(instance.Fcols):
                completion_bounds[i] += min(0, instance.C[i][j]) + min(0, instance.D[i][j])
        self.logger.debug("Completion bounds for follower constrs: {}".format(completion_bounds)) 

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
            self.update_completions_bounds(instance, completion_bounds, var_index, player)
            while len(current_layer_queue) and diagram.node_count < 1e8:
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

            if diagram.node_count >= 1e6:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if compilation == "restricted" and len(next_layer_queue) > max_width:
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

        clean_diagram.compilation_runtime = time() - t0

        self.logger.info("Diagram succesfully compiled. Time elapsed: {} - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, clean_diagram.node_count + 2, clean_diagram.arc_count, clean_diagram.width
        ))

        return clean_diagram

    def compile_diagram_LF(self, diagram, instance, compilation, max_width, ordering_heuristic):
        """
            This method compiles a DD, starting with the leader and continuing with the follower.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation type: {} - Compilation method: leader-follower".format(compilation))
        var_order = self.ordering_heuristic(instance, ordering_heuristic)
        self.logger.debug("Variable ordering: {}".format(var_order))
        n = instance.Lcols + instance.Fcols
        player = "leader"

        diagram.max_width = max_width
        diagram.compilation = compilation
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
            for j in range(instance.Fcols):
                completion_bounds[i] += min(0, instance.C[i][j]) + min(0, instance.D[i][j])
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

            if diagram.node_count >= 1e6:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if compilation == "restricted" and len(next_layer_queue) > max_width:
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

        clean_diagram.compilation_runtime = time() - t0

        self.logger.info("Diagram succesfully compiled. Time elapsed: {} - Node count: {} - Arc count: {} - Width: {}".format(
            time() - t0, clean_diagram.node_count + 2, clean_diagram.arc_count, clean_diagram.width
        ))

        return clean_diagram

    def compile_diagram_iteratively(self, diagram, instance, compilation, max_width, ordering_heuristic, Y):
        """
            This method compiles a DD, starting with the follower and continuing with the leader. 
            It only compiles y's belonging to set Y.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), Y (list), max_width (int)
            Returns: diagram (class DecisionDiagram)
        """

        t0 = time()
        self.logger.info("Compiling diagram. Compilation type: {} - Compilation method: iterative/collect_Y - MaxWidth: {}".format(compilation, max_width))
        var_order = self.ordering_heuristic(instance, ordering_heuristic)
        self.logger.debug("Variable ordering: {}".format(var_order))
        n = instance.Lcols + instance.Fcols

        diagram.max_width = max_width
        diagram.compilation = compilation
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
            for j in range(instance.Fcols):
                completion_bounds[i] += min(0, instance.C[i][j]) + min(0, instance.D[i][j])
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
                while len(current_layer_queue) and diagram.node_count < 1e6:
                    node = current_layer_queue.popleft()
                    if y[var_index] == 0:
                        child_node = self.create_zero_node(layer + 1, node)
                    else:
                        child_node = self.create_one_node(instance, layer + 1, var_index, node, player="follower")

                    # Zero head
                    if self.check_completion_bounds(instance, completion_bounds, child_node) and y[var_index] == 0:
                        child_node.id = diagram.node_count
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[child_node.hash_key]]
                            self.update_costs(node=found_node, new_node=child_node)
                            next_layer_queue.append(found_node)
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
                    
                    # One head
                    if self.check_completion_bounds(instance, completion_bounds, child_node) and y[var_index] == 1:
                        child_node.id = diagram.node_count
                        # Check if node was already created
                        if child_node.hash_key in diagram.graph_map:
                            found_node = diagram.nodes[diagram.graph_map[child_node.hash_key]]
                            self.update_costs(node=found_node, new_node=child_node)
                            next_layer_queue.append(found_node)
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
                        

                # Width limit
                if compilation == "restricted" and len(next_layer_queue) > max_width:
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
            while len(current_layer_queue) and diagram.node_count < 1e6:
                node = current_layer_queue.popleft()
                zero_head = self.create_zero_node(layer + 1, node)
                one_head = self.create_one_node(instance, layer + 1, var_index, node, player="leader")

                # Zero head
                if self.check_completion_bounds(instance, completion_bounds, zero_head):
                    zero_head.id = diagram.node_count
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
                    one_head.id = diagram.node_count
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

            if diagram.node_count >= 1e6:
                raise ValueError("Diagram surpassed max node count: {:e}".format(diagram.node_count))

            # Width limit
            if compilation == "restricted" and len(next_layer_queue) > max_width:
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

    def reduce_diagram(self, diagram):
        """
            This function executes the reduce algorithm by Bryant (1986).

            Args: diagram (class DecisionDiagram)
            Returns: None
        """

        t0 = time()
        self.logger.info("Executing reduce algorithm")

        # Populate nodes by layers (vlist)
        v_list = {layer: list() for layer in range(diagram.nodes["sink"].layer + 1)}
        for node in diagram.nodes.values():
            v_list[node.layer].append(node)

        # Traverse diagram bottom-up
        for layer in range(diagram.nodes["sink"].layer + 1)[::-1]:
            # Create keys for each node in current layer
            Q = list()
            for u in v_list[layer]:
                if u.id == "sink":
                    Q.append(((0, 0), u))
                else:
                    key = [float("inf"), float("inf")]
                    for arc in u.outgoing_arcs:
                        if arc.value == 0:
                            key[0] = arc.head
                        else:
                            key[1] = arc.head
                    Q.append((key, u))
            Q.sort(key=lambda x: x[0])

            # Merging process
            old_key = (-1, -1)
            for idx, (key, u) in enumerate(Q):
                self.logger.debug("Processing layer {} ({}/{})".format(layer, idx + 1, len(Q)))
                if key != old_key:
                    # Node cannot be merged with the previous one
                    old_key = key
                    old_node = u
                else:
                    # Node can be merged. Take each incoming arc and redirect its head
                    for in_arc in u.incoming_arcs:
                        in_arc.head = old_node.id
                        in_arc._update_id()
                        old_node.incoming_arcs.append(in_arc)
                    diagram.remove_node(u)
            if layer == diagram.nodes["sink"].layer - 1:
                # Special treatment for sink node
                diagram.nodes["sink"].incoming_arcs = [arc for arc in diagram.nodes["sink"].incoming_arcs if arc.tail in [old_node.id, "root"]]
        
        # Filter arcs
        diagram.arcs = [arc for arc in diagram.arcs if arc.tail in diagram.nodes and arc.head in diagram.nodes]
        for node in diagram.nodes.values():
            node.outgoing_arcs = [arc for arc in node.outgoing_arcs if arc.head in diagram.nodes]
            node.incoming_arcs = [arc for arc in node.incoming_arcs if arc.tail in diagram.nodes]

        diagram.reduce_algorithm_runtime = time() - t0

        self.logger.info("Diagram succesfully reduced. Time elapsed: {} sec. Nodes: {} - Arcs: {} - Width: {}".format(time() - t0, len(diagram.nodes), len(diagram.arcs), diagram.width))

    def ordering_heuristic(self, instance, ordering_heuristic):
        """
            This method retrieves a variable ordering.

            Args: instance (class Instance)
            Returns: dict of sorted indices (dict)
        """

        order = {"leader": list(), "follower": list()}

        # Sum coeffs of the left hand side
        if ordering_heuristic == "lhs_coeffs":
            self.logger.debug("Variable ordering heuristic: LHS coeffs")
            for j in range(instance.Fcols):
                coeffs_sum = sum(instance.B[i][j] + instance.D[i][j] for i in range(instance.Frows))
                order["follower"].append((j, coeffs_sum))
            for j in range(instance.Lcols):
                coeffs_sum = sum(instance.A[i][j] + instance.C[i][j] for i in range(instance.Frows))
                order["leader"].append((j, coeffs_sum))
            for key in order:  
                order[key].sort(key=lambda x: x[1])  # Sort variables in ascending order
                order[key] = [i[0] for i in order[key]]

        # Leader cost
        elif ordering_heuristic == "cost_leader":
            self.logger.debug("Variable ordering heuristic: leader cost")
            for j in range(instance.Fcols):
                order["follower"].append((j, instance.c_follower[j]))
            for j in range(instance.Lcols):
                order["leader"].append((j, instance.c_leader[j]))
            for key in order: 
                order[key].sort(key=lambda x: x[1])  # Sort variables in ascending order
                order[key] = [i[0] for i in order[key]]

        # Competitive costs
        elif ordering_heuristic == "cost_competitive":
            self.logger.debug("Variable ordering heuristic: leader and follower costs")
            for j in range(instance.Fcols):
                order["follower"].append((j, instance.d[j]))
            for j in range(instance.Lcols):
                order["leader"].append((j, instance.c_leader[j]))
            for key in order:
                order[key].sort(key=lambda x: x[1])  # Sort variables in ascending order
                order[key] = [i[0] for i in order[key]]

        # Leader feasibility
        elif ordering_heuristic == "leader_feasibility":
            self.logger.debug("Variable ordering heuristic: leader feasibility")
            for j in range(instance.Fcols):
                order["follower"].append((j, sum([instance.D[i][j] for i in range(instance.Frows)])))
            order["follower"].sort(key=lambda x: x[1])  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]
            for j in range(instance.Lcols):
                order["leader"].append((j, instance.c_leader[j]))
            order["leader"].sort(key=lambda x: x[1])  # Sort variables in ascending order
            order["leader"] = [i[0] for i in order["leader"]]
                
        
        else:
            self.logger.debug("Variable ordering heuristic: given order")
            order["follower"] = [i for i in range(instance.Fcols)]
            order["leader"] = [i for i in range(instance.Lcols)]

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
        """
            This method checks if any infeasible r-t path was compiled.

            Args: instance (class Instance), node (class Node).
            Returns: None
        """
        
        for i in range(instance.Frows):
            if node.state[i] > instance.b[i]:
                raise ValueError("Infeasible path. NodeID: {} - State: {}".format(node.id, node.state))
            
    def update_completions_bounds(self, instance, completion_bounds, var_index, player):
        for i in range(instance.Frows):
            if player == "follower":
                completion_bounds[i] -= min(0, instance.D[i][var_index])
            elif player == "leader":
                completion_bounds[i] -= min(0, instance.C[i][var_index])

    def update_costs(self, node, new_node):
        node.leader_cost = min(node.leader_cost, new_node.leader_cost)
        node.follower_cost = min(node.follower_cost, new_node.follower_cost)

    def reduced_queue(self, queue, max_width, player):
        if player == "follower":
            queue = sorted(queue, key=lambda x: int(0.9 * x.follower_cost + 0.1 * x.leader_cost))  # Sort nodes in ascending order # TODO: remove int
        else:
            queue = sorted(queue, key=lambda x: x.leader_cost)  # Sort nodes in ascending order
        
        return deque(queue[:max_width])

    def clean_diagram(self, diagram):
        t0 = time()
        self.logger.debug("Executing bottom-up recursion to remove unreachable nodes")
        
        clean_diagram = DecisionDiagram()
        clean_diagram.inherit_data(diagram)

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

        self.logger.debug("Process done. Time elapsed: {} sec".format(time() - t0))

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