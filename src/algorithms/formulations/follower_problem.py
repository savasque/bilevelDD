import gurobipy as gp

def run(instance, x):
    model = gp.Model()
    model.Params.OutputFlag = 0

    y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")

    model.addConstrs((gp.quicksum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) <= instance.b[i] for i in range(instance.Frows)), name="FollowerHPR")

    obj = gp.quicksum(instance.d[j] * y[j] for j in range(instance.Fcols))
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    model.optimize()

    return "Sol: {} - ObjVal: {}".format([y[i].X for i in y], model.ObjVal)