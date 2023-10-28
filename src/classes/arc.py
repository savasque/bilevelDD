class Arc:
    def __init__(self, tail, head, value, cost, var_index, player):
        self.id = "{}-{}-{}".format(tail, head, value)
        self.tail = tail
        self.head = head
        self.value = value
        self.cost = cost
        self.var_index = var_index
        self.player = player
    
    def __repr__(self):
        return self.id
    
    def _update_id(self):
        self.id = "{}-{}-{}".format(self.tail, self.head, self.value)