import gurobipy as gp

def run(instance, num_solutions):
    Y = list()
    while len(Y) <= num_solutions:
        Y += solve_follower_HPR(instance, Y, num_solutions=num_solutions)[1]
    
    return Y[:num_solutions]

def hamming_distance(y, y_2):
    return gp.quicksum(y[j] for j in y if y_2[j] == 0) + gp.quicksum(1 - y[j] for j in y if y_2[j] == 1)

def solve_follower_HPR(instance, forbidden_Y, num_solutions):
    model = gp.Model()
    model.Params.OutputFlag = 0

    model.Params.PoolSearchMode = 2  # Extra effort to find alternative optimal solutions
    model.Params.PoolSolutions = num_solutions

    x = model.addVars(instance.Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")

    # HPR constrs
    model.addConstrs(gp.quicksum(instance.A[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.B[i][j] * y[j] for j in range(instance.Fcols)) <= instance.a[i] for i in range(instance.Lrows))
    model.addConstrs(gp.quicksum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) <= instance.b[i] for i in range(instance.Frows))

    for y_2 in forbidden_Y:
        model.addConstr(hamming_distance(y, y_2) >= 1)

    # Objective function
    # obj_func = gp.quicksum(instance.D[i][j] * y[j] for i in range(instance.Frows) for j in range(instance.Fcols))
    obj_func = instance.d @ y.values()
    model.setObjective(obj_func, sense=gp.GRB.MINIMIZE)

    model.optimize()

    Y = dict()
    for i in range(num_solutions):
        model.Params.solutionNumber = i
        solution = model.getAttr("Xn")
        y = [int(solution[j]) for j in range(instance.Lcols, instance.Lcols + instance.Fcols)]
        Y[str(y)] = y
    
    Y = list(Y.values())
        
    return model.ObjVal, Y
