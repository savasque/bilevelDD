from docplex.mp.model import Model

def solve(instance, obj="leader", sense="minimize"):
    Lcols = instance.Lcols
    Fcols = instance.Fcols

    model = Model(log_output=False, float_precision=6)

    x = model.binary_var_dict(Lcols, name="x")
    y = model.binary_var_dict(Fcols, name="y")

    # HPR constrs
    model.addConstrs(model.sum(instance.A[i][j] * x[j] for j in range(instance.Lcols)) + model.sum(instance.B[i][j] * y[j] for j in range(instance.Fcols)) <= instance.a[i] for i in range(instance.Lrows))
    model.addConstrs(model.sum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) + model.sum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) <= instance.b[i] for i in range(instance.Frows))

    # Objective function
    if obj == "leader":
        obj_func = model.sum(instance.c_leader[j] * x[j] for j in range(Lcols)) + model.sum(instance.c_follower[j] * y[j] for j in range(Fcols))
    elif obj == "follower":
        obj_func = model.sum(instance.d[j] * y[j] for j in range(Fcols))
    if sense == "minimize":
        model.minimize(obj_func)
    else:
        model.maximize(obj_func)

    model.solve()

    vars = {
        "x": [x[i].solution_value for i in x],
        "y": [y[i].solution_value for i in y]
    }
    
    return model.ObjVal, vars