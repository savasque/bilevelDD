class Instance:
    def __init__(self, name, load_runtime, data):
        self.name = name
        self.load_runtime = load_runtime
        self.A = data["A"]
        self.B = data["B"]
        self.C = data["C"]
        self.D = data["D"]
        self.a = data["a"]
        self.b = data["b"]
        self.c_leader = data["c_leader"]
        self.c_follower = data["c_follower"]
        self.d = data["d"]
        self.Lcols = len(self.c_leader)
        self.Lrows = len(self.a)
        self.Fcols = len(self.c_follower)
        self.Frows = len(self.b)
        self.interaction = dict()
        self.known_y_values = dict()