from docplex.mp.model import Model

def get_model(instance, x, sense="minimize"):
    model = Model(log_output=False, float_precision=6)

    y = model.binary_var_dict(instance.Fcols, name="y")
    model._vars = {"y": y}

    constrs = {
        i: model.add_constraint(
            model.sum(instance.D[i][j] * y[j] for j in range(instance.Fcols))
            <= instance.b[i] - instance.C[i] @ x
        )
        for i in range(instance.Frows)
    }
    model._constrs = constrs

    obj = model.sum(instance.d[j] * y[j] for j in range(instance.Fcols))
    
    if sense == "minimize":
        model.minimize(obj)
    else:
        model.maximize(obj)

    return model