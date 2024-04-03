import gurobipy as gp

def solve(instance, x=None, sense="minimize"):
    model = gp.Model()
    model.Params.OutputFlag = 0

    if not x:
        x = model.addVars(instance.Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")

    model.addConstrs(
        (gp.quicksum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) 
         + gp.quicksum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) 
         <= instance.b[i] for i in range(instance.Frows)), 
         name="FollowerHPR"
    )

    obj = instance.d @ list(y.values())

    model.setObjective(obj, sense=gp.GRB.MINIMIZE if sense == "minimize" else gp.GRB.MAXIMIZE)

    model.optimize()

    return model.ObjVal, [y[i].X for i in y]