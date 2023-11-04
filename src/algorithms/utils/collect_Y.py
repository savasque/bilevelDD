from time import time

import gurobipy as gp

def run(instance, num_solutions):
    t0 = time()
    Y = dict()
    max_iters = 1e6
    iter = 0
    while len(Y) <= num_solutions and iter <= max_iters:
        new_ys = solve_follower_HPR(instance, Y, num_solutions=num_solutions, obj="follower")[1]
        for y in new_ys:
            Y[str(y)] = y
    # iter = 0
    # while len(Y) <= 2 * num_solutions // 3 and iter <= max_iters:
    #     new_ys = solve_follower_HPR(instance, Y, num_solutions=num_solutions, obj="leader")[1]
    #     for y in new_ys:
    #         Y[str(y)] = y
    # iter = 0
    # while len(Y) <= num_solutions // 2 and iter <= max_iters:
    #     new_ys = solve_follower_HPR(instance, Y, num_solutions=num_solutions, obj="leader_feasibility")[1]
    #     for y in new_ys:
    #         Y[str(y)] = y
    
    Y = list(Y.values())

    return Y[:num_solutions], time() - t0

def hamming_distance(y, y_2):
    return gp.quicksum(y[j] for j in y if y_2[j] == 0) + gp.quicksum(1 - y[j] for j in y if y_2[j] == 1)

def solve_follower_HPR(instance, forbidden_Y, num_solutions, obj):
    model = gp.Model()
    model.Params.OutputFlag = 0

    model.Params.PoolSearchMode = 2  # Extra effort to find alternative optimal solutions
    model.Params.PoolSolutions = num_solutions

    x = model.addVars(instance.Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")

    # HPR constrs
    model.addConstrs(gp.quicksum(instance.A[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.B[i][j] * y[j] for j in range(instance.Fcols)) <= instance.a[i] for i in range(instance.Lrows))
    model.addConstrs(gp.quicksum(instance.C[i][j] * x[j] for j in range(instance.Lcols)) + gp.quicksum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) <= instance.b[i] for i in range(instance.Frows))

    for y_2 in forbidden_Y.values():
        model.addConstr(hamming_distance(y, y_2) >= 1)

    # Objective function
    if obj == "leader":
        obj_func = instance.c_leader @ x.values() + instance.c_follower @ y.values()
    elif obj == "follower":
        obj_func = instance.c_leader @ x.values() + instance.d @ y.values()
    elif obj == "leader_feasibility":
        obj_func = gp.quicksum(((instance.D[i][j] + instance.B[i][j]) * y[j]) for i in range(instance.Frows) for j in range(instance.Fcols))
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
