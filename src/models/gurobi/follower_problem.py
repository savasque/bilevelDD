import gurobipy as gp

def get_model(instance):
    model = gp.Model()

    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"y": y}

    constrs = model.addConstr(instance.D @ y <= 0, name="FollowerHPR")
    model._constrs = constrs

    model.setObjective(instance.d @ y)

    model.update()

    return model