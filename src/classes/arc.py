class Arc:
    def __init__(self, tail, head, value, follower_cost, leader_cost, var_index, player):
        self.tail           = tail
        self.head           = head
        self.value          = value
        self.follower_cost  = follower_cost
        self.leader_cost    = leader_cost
        self.var_index      = var_index
        self.player         = player  # "leader" | "follower"

    @property
    def id(self):
        return  "{}-{}-{}".format(self.tail.id, self.head.id, self.value)
    
    def __repr__(self):
        return self.id