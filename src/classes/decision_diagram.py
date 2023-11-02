class DecisionDiagram:
    def __init__(self):
        self.nodes = dict()
        self.arcs = list()
        self.graph_map = dict()
        self.compilation = None
        self.max_width = None
        self.ordering_heuristic = None
        self.compilation_method = None
        self.var_order = None
        self.compilation_time = None
        self.reduce_algorithm_time = None
        self.initial_width = None

    @property
    def node_count(self):
        return len(self.nodes) - 2

    @property
    def arc_count(self):
        return len(self.arcs)
    
    @property
    def width(self):
        layers = self.nodes["sink"].layer
        width = 0
        for layer in range(layers):
            width = max(width, len([node for node in self.nodes.values() if node.layer == layer]))
        
        return width
    
    def add_node(self, node):
        self.graph_map[node.hash_key] = node.id
        self.nodes[node.id] = node

    def add_arc(self, arc):
        self.arcs.append(arc)

    def update_in_outgoing_arcs(self):
        for arc in self.arcs:
            self.nodes[arc.tail].outgoing_arcs.append(arc)
            self.nodes[arc.head].incoming_arcs.append(arc)

    def remove_in_outgoing_arcs(self):
        for node in self.nodes.values():
            node.incoming_arcs = list()
            node.outgoing_arcs = list()

    def is_solution_encoded(self, x, y):
        nodes = [self.nodes["root"]]
        while len(nodes):
            node = nodes.pop()
            if node.layer == len(x) + len(y):
                return True
            for arc in node.outgoing_arcs:  # Only for follower-leader compilation
                if (arc.player == "follower" and arc.value == y[self.var_order["follower"][node.layer]])\
                or (arc.player == "leader" and arc.value == x[self.var_order["leader"][node.layer - len(y)]]):
                    nodes.append(self.nodes[arc.head])
            if not nodes:
                a = None
        
        return False
    
    def inherit_data(self, diagram):
        self.compilation = diagram.compilation
        self.max_width = diagram.max_width
        self.ordering_heuristic = diagram.ordering_heuristic
        self.compilation_method = diagram.compilation_method
        self.var_order = diagram.var_order

    def remove_node(self, node):
        del self.nodes[node.id]
        
