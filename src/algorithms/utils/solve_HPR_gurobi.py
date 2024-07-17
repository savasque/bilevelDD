import gurobipy as gp

def solve(instance, obj="leader", sense="minimize", time_limit=3600):
    Lcols = instance.Lcols
    Fcols = instance.Fcols
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
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    x = model.addMVar(Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addMVar(Fcols, vtype=gp.GRB.BINARY, name="y")

    # HPR constrs
    if A.shape[0] != 0 and B.shape[0] != 0:
        model.addConstr((A @ x + B @ y <= a), name="LeaderHPR")
    elif A.shape[0] != 0:
        model.addConstr((A @ x <= a), name="LeaderHPR")
    elif B.shape[0] != 0:
        model.addConstr((B @ y <= a), name="LeaderHPR")
    model.addConstr((C @ x + D @ y <= b), name="FollowerHPR")

    # Objective function
    if obj == "leader":
        obj_func = c_leader @ x + c_follower @ y
    elif obj == "follower":
        obj_func = d @ y
    model.setObjective(obj_func, sense=gp.GRB.MINIMIZE if sense == "minimize" else gp.GRB.MAXIMIZE)

    model.optimize()

    vars = {
        "x": x.X,
        "y": y.X
    }
    
    return model.ObjVal, vars