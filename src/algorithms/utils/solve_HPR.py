import gurobipy as gp

def run(instance, obj="leader", sense="min"):
    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addVars(instance.Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")

    # HPR constrs
    model.addConstrs(gp.quicksum(instance.A[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.B[i][j] * y[j] for j in range(instance.Fcols)) <= instance.a[i] for i in range(instance.Lrows))
    model.addConstrs(gp.quicksum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) <= instance.b[i] for i in range(instance.Frows))

    # Objective function
    if obj == "leader":
        obj_func = instance.c_leader @ x.values() + instance.c_follower @ y.values()
    elif obj == "follower":
        obj_func = instance.d @ y.values()
    model.setObjective(obj_func, sense=gp.GRB.MINIMIZE if sense == "min" else gp.GRB.MAXIMIZE)

    model.optimize()

    vars = {
        "x": [x[i].X for i in x],
        "y": [y[i].X for i in y]
    }
    
    return model.ObjVal, vars