import gurobipy as gp

def get_model(instance):
    model = gp.Model()
    model.Params.OutputFlag = 0

    y = model.addMVar(instance.nF, vtype=gp.GRB.BINARY, name="y")
    z = model.addMVar(instance.mL, vtype=gp.GRB.BINARY, name="z")
    model._vars = {"y": y, "z": z}

    model.addConstr(gp.quicksum(1 - z[i] for i in range(instance.mL)) >= 1)

    model._constrs = {
        "vf_bound": model.addConstr(instance.d @ y <= 0, name="value_function"),
        "follower_constrs": model.addConstr(instance.D @ y <= 0, name="FollowerHPR"),
        "leader_blocking": model.addConstr(instance.B @ y + 1e8 * z >= 0, name="Blocking")
    }

    model.setObjective(0)

    model.update()

    return model