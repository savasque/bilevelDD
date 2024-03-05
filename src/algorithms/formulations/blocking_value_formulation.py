import gurobipy as gp

def get_model(instance, time_limit, Y):
    Lcols = instance.Lcols
    Lrows = instance.Lrows
    Fcols = instance.Fcols
    Frows = instance.Frows

    A = instance.A
    B = instance.B
    C = instance.C
    D = instance.D
    a = instance.a
    b = instance.b
    c_leader = instance.c_leader
    c_follower = instance.c_follower
    d = instance.d

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.IntegralityFocus = 1

    x = model.addVars(Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(Fcols, vtype=gp.GRB.BINARY, name="y")
    w = model.addVars(range(len(Y)), Frows, vtype=gp.GRB.BINARY, name="w")

    vars = {
        "x": x,
        "y": y,
        "w": w,
    }

    # HPR constrs
    model.addConstrs((A[i] @ x.values() + B[i] @ y.values() <= a[i] for i in range(Lrows)), name="LeaderHPR")
    model.addConstrs((C[i] @ x.values() + D[i] @ y.values() <= b[i] for i in range(Frows)), name="FollowerHPR")

    # Value-function constr
    M = 1e6
    for idx, sampled_y in enumerate(Y):
        model.addConstr(d @ y.values() <= d @ sampled_y - M * gp.quicksum(w[idx, i] for i in range(Frows)))

    # Blocking definition
    M = 1e6
    model.addConstrs(C[i] @ x.values() >= -M + gp.quicksum((M + b[i] - D[i] @ sampled_y) * w[idx, i] for idx, sampled_y in enumerate(Y)) for i in range(Frows))

    # Strengthening (Fischetti et al, 2017)
    model.addConstrs(y[j] == val for j, val in instance.known_y_values.items())

    # Objective function
    obj = c_leader @ x.values() + c_follower @ y.values()
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    return model, vars