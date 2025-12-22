import numpy as np

class Node:
    def __init__(self, id, layer, state=np.zeros(1), type=None):
        self.id = id
        self.state = state
        self.layer = layer
        self.outgoing_arcs = list()
        self.incoming_arcs = list()
        self.leader_cost = 0
        self.follower_cost = 0
        self.type = type
    
    @property
    def hash_key(self):
        return "{}-{}".format(
            np.array2string(
                self.state,
                formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:g}"},
            ).replace("\n", ""),
            self.layer
        )
    
    def __repr__(self):
        return "{}-{}".format(self.id, self.layer)
    
    def inherit_data(self, node):
        self.outgoing_arcs = node.outgoing_arcs
        self.incoming_arcs = node.incoming_arcs
        self.leader_cost = node.leader_cost
        self.follower_cost = node.follower_cost
