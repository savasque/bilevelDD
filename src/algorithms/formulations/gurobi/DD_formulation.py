from time import time
import gurobipy as gp
import numpy as np

def get_model(instance, diagram, incumbent=None):
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

    t0 = time()
    model = gp.Model()

    x = model.addMVar(Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addMVar(Fcols, vtype=gp.GRB.BINARY, name="y")

    if incumbent:
        for j in range(Lcols):
            x[j].start = incumbent["x"][j]
        for j in range(Fcols):
            y[j].start = incumbent["y"][j]

    model._vars = {
        "x": x,
        "y": y
    }

    # HPR constrs
    if A.shape[0] != 0 and B.shape[0] != 0:
        model.addConstr((A @ x + B @ y <= a), name="LeaderHPR")
    elif A.shape[0] != 0:
        model.addConstr((A @ x <= a), name="LeaderHPR")
    elif B.shape[0] != 0:
        model.addConstr((B @ y <= a), name="LeaderHPR")
    model.addConstr((C @ x + D @ y <= b), name="FollowerHPR")

    # Objective function
    obj = c_leader @ x + c_follower @ y
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    if diagram:
        nodes = diagram.nodes
        arcs = diagram.arcs
        root_node = diagram.root_node
        sink_node = diagram.sink_node
        interaction_rows = [i for i in range(instance.Frows) if instance.interaction[i] == "both"]

        pi = model.addVars([node.id for node in nodes], lb=-gp.GRB.INFINITY, name="pi")
        lamda = model.addVars([arc.id for arc in arcs if arc.player == "leader"], name="lambda")
        gamma = model.addVars([arc.id for arc in arcs if arc.player == "leader"], name="gamma")
        alpha = model.addVars([arc.id for arc in arcs if arc.player == "leader"], vtype=gp.GRB.BINARY, name="alpha")
        beta = model.addVars([arc.id for arc in arcs if arc.player == "leader"], interaction_rows, vtype=gp.GRB.BINARY, name="beta")

        model._vars.update(
            {
                "pi": pi,
                "lambda": lamda,
                "gamma": gamma,
                "alpha": alpha,
                "beta": beta
            }
        )

        # Value-function constr
        model.addConstr(d @ y <= pi[root_node.id], name="ValueFunction")

        # Dual feasibility
        model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] <= arc.cost for arc in arcs if arc.player in ["follower", "dummy"]), name="DualFeas0")
        model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] - lamda[arc.id] <= arc.cost for arc in arcs if arc.player == "leader"), name="DualFeas1")
        
        # Strong duality
        model.addConstr(pi[sink_node.id] == 0, name="StrongDualSink")

        # Gamma bounds
        M = 1e6
        model.addConstrs(gamma[arc.id] <= (M - arc.tail.follower_cost) * alpha[arc.id] for arc in arcs if arc.player == "leader")

        # Alpha-beta relationship
        model.addConstrs(alpha[arc.id] <= gp.quicksum(beta[arc.id, i] for i in interaction_rows) for arc in arcs if arc.player == "leader")

        # Blocking definition
        M = {i: sum(min(C[i][j], 0) for j in range(Lcols)) for i in interaction_rows}
        model.addConstrs(
            C[i] @ x >= M[i] + beta[arc.id, i] * (-M[i] + instance.b[i] - arc.tail.state[i] + 1) 
            for arc in arcs if arc.player == "leader" for i in interaction_rows
        )

    return model, time() - t0