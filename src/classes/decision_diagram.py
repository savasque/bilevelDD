class DecisionDiagram:
    def __init__(self):
        self.nodes = dict()
        self.arcs = list()
        self.graph_map = dict()

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