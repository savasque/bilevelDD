from time import time

import gurobipy as gp

def get_model(instance):
    t0 = time()

    model = gp.Model()

    x = model.addMVar(instance.nL, vtype=gp.GRB.BINARY, name="x")
    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"x": x, "y": y}

    if instance.mL > 0:
        model.addConstr(instance.A @ x + instance.B @ y <= instance.a, name="LeaderConstrs")
    model.addConstr(instance.C @ x + instance.D @ y <= instance.b, name="FollowerHPR")

    model.setObjective(instance.cL @ x + instance.cF @ y)

    model.update()

    model._build_time = time() - t0

    return model