import gurobipy as gp

def solve(instance, x, objval):
    model = gp.Model()
    model.Params.OutputFlag = 0

    y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")

    model.addConstrs(gp.quicksum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) <= instance.b[i] for i in range(instance.Frows))

    model.addConstr(gp.quicksum(instance.d[j] * y[j] for j in range(instance.Fcols)) == objval)

    obj = gp.quicksum(instance.c_follower[j] * y[j] for j in range(instance.Fcols))
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    model.optimize()

    return model.ObjVal, [y[i].X for i in y]