class Arc:
    def __init__(self, tail, head, value, cost, var_index, player):
        self.tail = tail
        self.head = head
        self.value = value
        self.cost = cost
        self.var_index = var_index
        self.player = player  # "leader" | "follower" | "dummy"
        self.block_values = dict()

    @property
    def id(self):
        return  "{}-{}-{}".format(self.tail.id, self.head.id, self.value)
    
    def __repr__(self):
        return self.id