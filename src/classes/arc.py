class Arc:
    def __init__(self, tail, head, value, cost, var_index, player):
        self.tail = tail
        self.head = head
        self.value = value
        self.cost = cost
        self.var_index = var_index
        self.player = player