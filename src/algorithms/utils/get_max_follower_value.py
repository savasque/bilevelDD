import gurobipy as gp

def get_max_follower_value(instance):
    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addMVar(instance.nL, vtype=gp.GRB.BINARY, name="x")
    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")

    model.addConstr(instance.C @ x + instance.D @ y <= instance.b)

    obj = instance.d @ y

    model.setObjective(obj, sense=gp.GRB.MAXIMIZE)

    model.optimize()

    return model.ObjVal