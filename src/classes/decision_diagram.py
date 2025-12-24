class DecisionDiagram:
    def __init__(self, id, args):
        self.id                         = id
        self.encoding                   = args["encoding"]
        self.max_width                  = args["max_width"]  # Max width allowed when compiling the DD
        self.ordering_heuristic         = args["ordering_heuristic"]  # Variable ordering
        self.reduce_method              = args["reduce_method"]
        self.problem_setting            = args["problem_setting"]   # Optimistic or pessimistic 
        self.nodes                      = list()
        self.arcs                       = list()
        self.root_node                  = None
        self.sink_node                  = None
        self.graph_map                  = dict()  # {hash_key: Node}: Hast table mapping hash_key (node.state + node.layer) to an instance of Node
        self.var_order                  = None  # {position (int): Node}
        self.compilation_runtime        = 0  # Total time for retrieving the final DD
        self.reduce_algorithm_runtime   = 0  # Time elapsed during reduction
        self.sampling_runtime           = 0  # Time elapsed during sampling
        self.initial_width              = 0  # DD width before reduction

    @property
    def node_count(self):
        return len(self.nodes)

    @property
    def arc_count(self):
        return len(self.arcs)
    
    @property
    def width(self):
        return max(len(self.graph_map[layer]) for layer in range(self.sink_node.layer))
    
    @property
    def num_merges(self):
        return len([node for node in self.nodes if node.id != -1 and len(node.incoming_arcs) >= 2])
    
    def add_node(self, node):
        self.graph_map[node.layer][node.hash_key] = node
        self.nodes.append(node)
        if node.id == 0:
            self.root_node = node
        elif node.id == -1:
            self.sink_node = node

    def add_arc(self, arc, update_in_outgoing_arcs=True):
        self.arcs.append(arc)
        if update_in_outgoing_arcs:
            arc.tail.outgoing_arcs.append(arc)
            arc.head.incoming_arcs.append(arc)

    def update_in_outgoing_arcs(self):
        for arc in self.arcs:
            if arc not in self.graph_map[arc.tail.hash_key].outgoing_arcs:
                self.graph_map[arc.tail.hash_key].outgoing_arcs.append(arc)
            if arc not in self.graph_map[arc.head.hash_key].incoming_arcs:
                self.graph_map[arc.head.hash_key].incoming_arcs.append(arc)

    def remove_in_outgoing_arcs(self):
        for node in self.nodes:
            node.incoming_arcs = list()
            node.outgoing_arcs = list()

    def is_solution_encoded(self, x, y):
        """
            This method indicates whether a solution (x, y) is encoded in the diagram. Only for follower-then-leader type of compilation
        """
        
        nodes = [self.root_node]
        while len(nodes):
            node = nodes.pop()
            if node.layer == len(x) + len(y):
                return True
            for arc in node.outgoing_arcs:  # Only for follower-compressed-leader compilation
                if (arc.player == "follower" and arc.value == y[self.var_order["follower"][node.layer]]) or arc.player == "leader":
                    nodes.append(arc.head)
        
        return False
    
    def inherit_data(self, diagram):
        self.id = diagram.id
        self.max_width = diagram.max_width
        self.ordering_heuristic = diagram.ordering_heuristic
        self.compilation_method = diagram.compilation_method
        self.var_order = diagram.var_order
        
