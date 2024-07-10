from docplex.mp.model import Model

def get_model(instance, x, objval):
    model = Model(log_output=False, float_precision=6)

    y = model.binary_var_dict(instance.Fcols, name="y")
    model._vars = {"y": y}

    model._constrs = {
        "HPR": {
            i: model.add_constraint(
                model.sum(instance.D[i][j] * y[j] for j in range(instance.Fcols))
                <= instance.b[i] - instance.C[i] @ x
            )
            for i in range(instance.Frows)
        },
        "objval": model.add_constraint(model.sum(instance.d[j] * y[j] for j in range(instance.Fcols)) <= objval)
    }
    
    obj = model.sum(instance.c_follower[j] * y[j] for j in range(instance.Fcols))
    model.minimize(obj)

    return model