class Node:
    def __init__(self, id, layer, state=None):
        self.id = id
        self.state = state
        self.layer = layer
        self.outgoing_arcs = list()
        self.incoming_arcs = list()
        self.leader_cost = 0
        self.follower_cost = 0
    
    @property
    def hash_key(self):
        return "{}-{}".format(self.state, self.layer)
    
    def __repr__(self):
        return "{}-{}".format(self.id, self.hash_key)
