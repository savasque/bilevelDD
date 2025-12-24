import numpy as np

class Node:
    def __init__(self, id, layer, state=np.array([None]), follower_cost=None, leader_cost=None, type=None):
        self.id             = id
        self.state          = state
        self.layer          = layer
        self.outgoing_arcs  = list()
        self.incoming_arcs  = list()
        self.follower_cost  = follower_cost
        self.leader_cost    = leader_cost
        self.type           = type
    
    @property
    def hash_key(self):
        return "{}-{}".format(
            self.state.tobytes(),
            self.layer
        )
    
    def __repr__(self):
        return "{}-{}".format(self.id, self.layer)
    
    def inherit_data(self, node):
        self.outgoing_arcs = node.outgoing_arcs
        self.incoming_arcs = node.incoming_arcs
        self.leader_cost = node.leader_cost
        self.follower_cost = node.follower_cost
