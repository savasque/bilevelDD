from time import time
from collections import deque
from random import shuffle
import numpy as np

from classes.node import Node


class Operations:
    def __init__(self, logger):
        self.logger = logger

    def reduce_diagram(self, diagram):
        """
            This function executes the reduce algorithm by Bryant (1986).

            Args: diagram (class DecisionDiagram)
            Returns: None
        """

        t0 = time()
        self.logger.info("Executing reduce algorithm")

        # Populate nodes by layers (vlist)
        v_list = {layer: list() for layer in range(diagram.sink_node.layer + 1)}
        for node in diagram.nodes:
            v_list[node.layer].append(node)

        # Traverse diagram bottom-up
        for layer in range(diagram.sink_node.layer + 1)[::-1]:
            # if diagram.compilation_method == "follower_then_compressed_leader" and layer == diagram.nodes["sink"].layer - 1:
            #     continue
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

            # Merge nodes with equivalent keys
            old_key = (-1, -1)
            for idx, (key, u) in enumerate(Q):
                # self.logger.debug("Processing layer {} ({}/{})".format(layer, idx + 1, len(Q)))
                if key != old_key:
                    # Node cannot be merged with the previous one
                    old_key = key
                    old_node = u
                else:
                    # Node can be merged. Take each incoming arc and redirect their heads
                    for in_arc in u.incoming_arcs:
                        in_arc.head = int(old_node.id)
                        old_node.incoming_arcs.append(in_arc)
                    u.incoming_arcs = list()
                    u.outgoing_arcs = list()
            if layer == diagram.sink_node.layer - 1:
                # Special treatment for sink node
                diagram.sink_node.incoming_arcs = [arc for arc in diagram.sink_node.incoming_arcs if arc.tail in [old_node.id, "root"]]
        
        # Filter arcs
        diagram.arcs = [arc for arc in diagram.arcs if arc.tail in diagram.nodes and arc.head in diagram.nodes]
        for node in diagram.nodes.values():
            node.outgoing_arcs = [arc for arc in node.outgoing_arcs if arc.head in diagram.nodes]
            node.incoming_arcs = [arc for arc in node.incoming_arcs if arc.tail in diagram.nodes]

        diagram.reduce_algorithm_runtime = time() - t0

        self.logger.info("Diagram succesfully reduced. Time elapsed: {} s - Nodes: {} - Arcs: {} - Width: {}".format(time() - t0, len(diagram.nodes), len(diagram.arcs), diagram.width))

    def ordering_heuristic(self, instance, ordering_heuristic, compressed_leader=False):
        """
            This method retrieves a variable ordering.

            Args: instance (class Instance)
            Returns: dict of sorted indices (dict)
        """
        
        order = {"leader": list(), "follower": list()}
        t0 = time()

        # Sum coeffs of the left hand side
        if ordering_heuristic == "lhs_coeffs":
            self.logger.debug("Variable ordering heuristic: LHS coeffs")
            for j in range(instance.Fcols):
                coeffs_sum = sum(instance.D[i][j] for i in range(instance.Frows) if instance.interaction[i] == "both")
                order["follower"].append((j, coeffs_sum))
            order["follower"].sort(key=lambda x: x[1])  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]

        # Leader cost
        elif ordering_heuristic == "leader_cost":
            self.logger.debug("Variable ordering heuristic: leader cost")
            for j in range(instance.Fcols):
                order["follower"].append((j, instance.c_follower[j]))
            order["follower"].sort(key=lambda x: x[1])  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]

        # Follower costs
        elif ordering_heuristic == "follower_cost":
            self.logger.debug("Variable ordering heuristic: follower cost")
            order["follower"] = sorted([j for j in range(instance.Fcols)], key=lambda x: instance.d[x])

        # Leader feasibility
        elif ordering_heuristic == "leader_feasibility":
            self.logger.debug("Variable ordering heuristic: leader feasibility")
            for j in range(instance.Fcols):
                order["follower"].append((j, max([instance.D[i][j] for i in range(instance.Frows)]), sum([instance.D[i][j] for i in range(instance.Frows)])))
            order["follower"].sort(key=lambda x: (x[1], x[2]))  # Sort variables in ascending order
            order["follower"] = [i[0] for i in order["follower"]]
                
        # Lexicographic
        elif ordering_heuristic == "lexicographic":
            self.logger.debug("Variable ordering heuristic: lexicographic")
            order["follower"] = [i for i in range(instance.Fcols)]
            if not compressed_leader:
                order["leader"] = [i for i in range(instance.Lcols)]

        # Max-connected-degree
        elif ordering_heuristic == "max_connected_degree":
            # Select follower variables with known value
            order["follower"] = [j for j in instance.known_y_values]
            remaining_columns = [j for j in range(instance.Fcols) if j not in instance.known_y_values]

            # Select follower variable with more appearances in D together with the already included variables
            degree_sequence = {j: 0 for j in remaining_columns}
            for i in range(instance.Frows):
                for j in remaining_columns:
                    if instance.D[i][j] != 0 and instance.interaction[i] == "follower": 
                        degree_sequence[j] += 1
                        for k in instance.known_y_values:
                            if instance.D[i][k] != 0:
                                degree_sequence[j] += 1
            order["follower"].append(max(degree_sequence.items(), key=lambda x: x[1])[0])
            remaining_columns = [i for i in range(instance.Fcols) if i != order["follower"][-1]]

            # Select follower variable with the most appearances in D where the last column also appears
            while remaining_columns:
                degree_sequence = {j: [None, 0, 0] for j in remaining_columns}  # [idx, joint appearances, degree]
                last_column = order["follower"][-1]
                for i in range(instance.Frows):
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
        
        order["leader"] = [i for i in range(instance.Lcols)]

        return order, time() - t0

    def create_zero_node(self, layer, parent_node):
        node = Node(id=None, layer=layer, state=list(parent_node.state), type=0)
        node.leader_cost = float(parent_node.leader_cost)
        node.follower_cost = float(parent_node.follower_cost)
        
        return node

    def create_one_node(self, instance, layer, var_index, parent_node, player):
        node = Node(id=None, layer=layer, state=list(parent_node.state), type=1)
        if player == "follower":
            for i in range(instance.Frows):
                node.state[i] += instance.D[i][var_index]
            node.leader_cost = parent_node.leader_cost + instance.c_follower[var_index]
            node.follower_cost = parent_node.follower_cost + instance.d[var_index]
        else:
            for i in range(instance.Frows):
                node.state[i] += instance.C[i][var_index]
            node.leader_cost = parent_node.leader_cost + instance.c_leader[var_index]
            node.follower_cost = float(parent_node.follower_cost)

        return node

    def check_completion_bounds(self, instance, completion_bounds, node):           
        if np.any([node.state + completion_bounds > instance.b]):
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
                completion_bounds[i] -= min(0, instance.D[i, var_index])
            elif player == "leader":
                completion_bounds[i] -= min(0, instance.C[i, var_index])

    def update_costs(self, node, new_node):
        node.leader_cost = min(node.leader_cost, new_node.leader_cost)
        node.follower_cost = min(node.follower_cost, new_node.follower_cost)

    def reduce_queue(self, instance, discard_method, queue, max_width):
        if discard_method == "follower_cost":
            sorted_queue = deque(sorted(queue, key=lambda x: int(0.9 * x.follower_cost + 0.1 * x.leader_cost)))
        elif discard_method == "minmax_state":
            sorted_queue = deque(sorted(queue, key=lambda x: (max(x.state - instance.b), sum(x.state), x.follower_cost)))
        elif discard_method == "random":
            sorted_queue = queue
            shuffle(sorted_queue)
        elif discard_method == "minsum_state":
            sorted_queue = deque(sorted(queue, key=lambda x: sum(x.state), reverse=True))

        filtered_queue = deque()
        remaining_nodes = deque()
        
        # Check diversity of the queue
        for node in sorted_queue:
            if self.check_diversity_criterion(instance, filtered_queue, node) and len(filtered_queue) < max_width:
                filtered_queue.append(node)
            else:
                remaining_nodes.append(node)
        
        # Replenish queue if too small
        while len(filtered_queue) < max_width:
            filtered_queue.append(remaining_nodes.popleft())
        
        return filtered_queue

    def check_diversity_criterion(self, instance, nodes, new_node):
        for node in nodes:
            if self.blocking_distance(instance, new_node, node) <= 0:
                return False
        
        return True
    
    def blocking_distance(self, instance, node_1, node_2):
        distance = 0
        for i in range(instance.Frows):
            if instance.interaction[i] == "both":
                if node_1.state[i] < node_2.state[i]:
                    distance += 1

        return distance

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

        self.logger.debug("Bottom-up filtering done -> Time elapsed: {} s".format(round(time() - t0)))

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
