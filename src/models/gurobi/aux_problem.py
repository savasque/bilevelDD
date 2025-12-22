import gurobipy as gp

def get_model(instance):
    model = gp.Model()

    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"y": y}

    model._constrs = {
        "follower_constrs": model.addConstr(instance.D @ y <= 0),
        "vf_bound": model.addConstr(instance.d @ y <= 0)
    }
    
    model.setObjective(instance.cF @ y)

    model.update()

    return model