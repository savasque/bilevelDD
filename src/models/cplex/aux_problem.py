from docplex.mp.model import Model

def get_model(instance):
    model = Model(log_output=False, float_precision=6)

    y = model.binary_var_dict(instance.nF, name="y")
    model._vars = {"y": y}

    model._constrs = {
        "follower_constrs": {
            i: model.add_constraint(
                model.sum(instance.D[i][j] * y[j] for j in range(instance.mF))
                <= 0
            )
            for i in range(instance.mF)
        },
        "vf_bound": model.add_constraint(model.sum(instance.d[j] * y[j] for j in range(instance.nF)) <= 0)
    }
    
    obj = model.sum(instance.cF[j] * y[j] for j in range(instance.nF))
    model.minimize(obj)

    return model