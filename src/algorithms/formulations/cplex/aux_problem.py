import gurobipy as gp

def get_model(instance, x, objval):
    model = gp.Model()
    model.Params.OutputFlag = 0

    y = model.addMVar(instance.Fcols, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"y": y}

    model._constrs = {
        "HPR": model.addConstr(
            instance.D @ y
            <= instance.b - instance.C @ x
        ),
        "objval": model.addConstr(instance.d @ y <= objval)
    }
    
    obj = instance.c_follower @ y
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    return model