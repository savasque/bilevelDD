import sys
sys.setrecursionlimit(2000)  #  TODO: rewrite the recursive filtering method in an iterative fashion

from time import time
from collections import deque
import numpy as np

from classes.node import Node
from classes.decision_diagram import DecisionDiagram


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
        v_list = {layer: list() for layer in range(diagram.nodes["sink"].layer + 1)}
        for node in diagram.nodes.values():
            v_list[node.layer].append(node)

        # Traverse diagram bottom-up
        for layer in range(diagram.nodes["sink"].layer + 1)[::-1]:
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

        self.logger.info("Diagram succesfully reduced. Time elapsed: {} s - Nodes: {} - Arcs: {} - Width: {}".format(time() - t0, len(diagram.nodes), len(diagram.arcs), diagram.width))

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
                coeffs_sum = sum(instance.D[i][j] for i in range(instance.Frows) if instance.interaction[i] == "both")
                order["follower"].append((j, coeffs_sum))
            for j in range(instance.Lcols):
                coeffs_sum = sum(instance.C[i][j] for i in range(instance.Frows))
                order["leader"].append((j, coeffs_sum))
            for key in order:  
                order[key].sort(key=lambda x: x[1])  # Sort variables in ascending order
                order[key] = [i[0] for i in order[key]]

        # Leader cost
        elif ordering_heuristic == "leader_cost":
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
            self.logger.debug("Variable ordering heuristic: leader and follower (competitive) costs")
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
                
        elif ordering_heuristic == "lexicographic":
            self.logger.debug("Variable ordering heuristic: lexicographic")
            order["follower"] = [i for i in range(instance.Fcols)]
            order["leader"] = [i for i in range(instance.Lcols)]

        elif ordering_heuristic == "max_connected_degree":
            # Select follower variable with the most appearances in D
            degree_sequence = {j: 0 for j in range(instance.Fcols)}
            for i in range(instance.Frows):
                for j in range(instance.Fcols):
                    if instance.D[i][j] != 0:
                        if instance.interaction[i] == "follower":  # Only follower constrs
                            degree_sequence[j] += 1
            order["follower"].append(max(degree_sequence.items(), key=lambda x: x[1])[0])
            remaining_columns = [i for i in range(instance.Fcols) if i != order["follower"][0]]

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
            order["leader"] = [i for i in range(instance.Lcols)]

        return order

    def create_zero_node(self, layer, parent_node):
        node = Node(id=None, layer=layer, state=list(parent_node.state))
        node.leader_cost = float(parent_node.leader_cost)
        node.follower_cost = float(parent_node.follower_cost)

        return node

    def create_one_node(self, instance, layer, var_index, parent_node, player):
        node = Node(id=None, layer=layer, state=list(parent_node.state))
        if player == "follower":
            for i in range(instance.Frows):
                if node.state[i] != None:
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
        for i in range(instance.Frows):
            if node.state[i] != None and node.state[i] + completion_bounds[i] > instance.b[i]:
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
            if completion_bounds[i] != None:
                if player == "follower":
                    completion_bounds[i] -= min(0, instance.D[i][var_index])
                elif player == "leader":
                    completion_bounds[i] -= min(0, instance.C[i][var_index])

    def update_costs(self, node, new_node):
        node.leader_cost = min(node.leader_cost, new_node.leader_cost)
        node.follower_cost = min(node.follower_cost, new_node.follower_cost)

    def reduced_queue(self, instance, queue, max_width, player):
        if player == "follower":
            sorted_queue = deque(sorted(queue, key=lambda x: int(0.9 * x.follower_cost + 0.1 * x.leader_cost)))  # Sort nodes in ascending order # TODO: remove int
        else:
            sorted_queue = deque(sorted(queue, key=lambda x: x.leader_cost))  # Sort nodes in ascending order

        filtered_queue = deque([sorted_queue.popleft()])
        for node in sorted_queue:
            if self.check_diversity_criterion(instance, filtered_queue, node):
                filtered_queue.append(node)
            if len(filtered_queue) == max_width:
                break
        
        return filtered_queue

    def check_diversity_criterion(self, instance, nodes, new_node):
        for node in nodes:
            if self.blocking_distance(instance, new_node, node) <= 0:
                return False
        
        return True
    
    def blocking_distance(self, instance, node_1, node_2):
        distance = 0
        for i in range(instance.Frows):
            if node_1.state[i] != None and instance.interaction[i] == "both" and node_1.state[i] < node_2.state[i]:
                distance += 1

        return distance

    def clean_diagram(self, diagram):
        t0 = time()
        self.logger.debug("Executing bottom-up recursion to remove unreachable nodes")
        
        clean_diagram = DecisionDiagram()
        clean_diagram.inherit_data(diagram)

        # Add root and sink nodes
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

        self.logger.debug("Bottom-up recursion done. Time elapsed: {} s".format(time() - t0))

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