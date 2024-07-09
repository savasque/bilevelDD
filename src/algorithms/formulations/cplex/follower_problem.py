import gurobipy as gp

def get_model(instance, x, sense="minimize"):
    model = gp.Model()
    model.Params.OutputFlag = 0

    y = model.addMVar(instance.Fcols, vtype=gp.GRB.BINARY, name="y")
    model._vars = {"y": y}

    constrs = model.addConstr(
        (instance.D @ y <= instance.b - instance.C @ x), 
         name="FollowerHPR"
    )
    model._constrs = constrs

    obj = instance.d @ y

    model.setObjective(obj, sense=gp.GRB.MINIMIZE if sense == "minimize" else gp.GRB.MAXIMIZE)

    return model