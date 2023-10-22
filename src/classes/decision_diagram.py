class DecisionDiagram:
    def __init__(self):
        self.nodes = dict()
        self.arcs = list()
        self.graph_map = dict()
        self.last_added_node = None

    @property
    def node_count(self):
        return len(self.nodes) - 2

    @property
    def arc_count(self):
        return len(self.arcs)
    
    def add_node(self, node):
        self.graph_map[node.hash_key] = node.id
        self.nodes[node.id] = node
        if node.id not in ["root", "sink"]:
            self.last_added_node = node.id

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