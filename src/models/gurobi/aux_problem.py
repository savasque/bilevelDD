import gurobipy as gp

def get_model(instance):
    model = gp.Model()

    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"y": y}

    model._constrs = {
        "leader_constrs": model.addConstr(instance.B @ y <= float("inf")),
        "follower_constrs": model.addConstr(instance.D @ y <= float("inf")),
        "vf_bound": model.addConstr(instance.d @ y <= float("inf")),
    }
    
    model.setObjective(instance.cF @ y)

    model.update()

    return model