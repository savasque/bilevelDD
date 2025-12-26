import gurobipy as gp

def get_model(instance):
    model = gp.Model()
    model.Params.OutputFlag = 0

    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"y": y}

    model._constrs = {
        "follower_constrs": model.addConstr((instance.D @ y <= 0), name="FollowerHPR"),
        "vf_bound": model.addConstr((instance.d @ y <= 0), name="FollowerObjVal")
    }

    obj = instance.cF @ y

    model.setObjective(obj, sense=gp.GRB.MAXIMIZE)

    model.update()

    return model