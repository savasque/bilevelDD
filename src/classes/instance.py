class Instance:
    def __init__(self, name, load_runtime, data):
        self.name = name
        self.load_runtime = load_runtime
        self.problem_type = data["problem_type"]
        self.A = data["A"]
        self.B = data["B"]
        self.C = data["C"]
        self.D = data["D"]
        self.a = data["a"]
        self.b = data["b"]
        self.cL = data["cL"]
        self.cF = data["cF"]
        self.d = data["d"]
        self.nL = data["nL"]
        self.mL = data["mL"]
        self.nF = data["nF"]
        self.mF = data["mF"]
        self.interaction = dict()
        self.known_y_values = dict()
        self.graph = data.get("graph")  # Only for BISP-KC instances
        self.p = data.get("p")  # Only for BISP-KC instances