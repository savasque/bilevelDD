from time import time
import gurobipy as gp
import numpy as np

def get_model(instance, diagram, max_follower_value, incumbent=None):
    nL = instance.nL
    mL = instance.mL
    nF = instance.nF
    A = instance.A
    B = instance.B
    C = instance.C
    D = instance.D
    a = instance.a
    b = instance.b
    cL = instance.cL
    cF = instance.cF
    d = instance.d

    t0 = time()
    model = gp.Model()
    model.Params.NonConvex = 2

    x = model.addMVar(nL, vtype=gp.GRB.BINARY, name="x")
    y = model.addMVar(nF, vtype=gp.GRB.BINARY, name="y")

    if incumbent:
        for j in range(nL):
            x[j].start = incumbent["x"][j]
        for j in range(nF):
            y[j].start = incumbent["y"][j]

    model._vars = {
        "x": x,
        "y": y
    }

    # HPR constrs
    if mL > 0:
        model.addConstr(A @ x + B @ y <= a, name="LeaderHPR")
    model.addConstr(C @ x + D @ y <= b, name="FollowerHPR")

    # Objective function
    model.setObjective(cL @ x + cF @ y)

    if diagram:
        nodes = diagram.nodes
        arcs = diagram.arcs
        root_node = diagram.root_node
        sink_node = diagram.sink_node

        pi = model.addVars([node.id for node in nodes], lb=-gp.GRB.INFINITY, name="pi")
        # lamda = model.addVars(Lcols, name="lambda")
        # beta = model.addVars(Lcols, name="beta")
        lamda = model.addVars([arc.id for arc in arcs if arc.value == 0], name="lambda")
        beta = model.addVars([arc.id for arc in arcs if arc.value == 1], name="beta")

        model._vars.update(
            {
                "pi": pi,
                "lambda": lamda,
                "beta": beta
            }
        )

        # Value-function constr
        model.addConstr(d @ y <= pi[root_node.id], name="ValueFunction")

        # Dual feasibility
        model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] <= arc.cost for arc in arcs if arc.player == "follower"), name="DualFeas0")
        
        # model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] - lamda[arc.var_index] <= 0 for arc in arcs if arc.player == "leader" and arc.value == 0), name="DualFeas1")
        # model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] - beta[arc.var_index] <= 0 for arc in arcs if arc.player == "leader" and arc.value == 1), name="DualFeas2")
        model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] - lamda[arc.id] <= 0 for arc in arcs if arc.player == "leader" and arc.value == 0), name="DualFeas1")
        model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] - beta[arc.id] <= 0 for arc in arcs if arc.player == "leader" and arc.value == 1), name="DualFeas2")
        
        model.addConstr(pi[sink_node.id] == 0, name="StrongDualSink")

        # Primal-dual linearization
        M = 1e6
        # model.addConstrs(lamda[j] <= M * x[j] for j in range(Lcols))
        # model.addConstrs(beta[j] <= M * (1 - x[j]) for j in range(Lcols))
        model.addConstrs(lamda[arc.id] <= M * x[arc.var_index] for arc in arcs if arc.value == 0)
        model.addConstrs(beta[arc.id] <= M * (1 - x[arc.var_index]) for arc in arcs if arc.value == 1) 
        
        # model.addConstrs(lamda[j] * (1 - x[j]) == 0 for j in range(Lcols))
        # model.addConstrs(beta[j] * x[j] == 0 for j in range(Lcols))

        # # Flow mapping
        # w = model.addVars([arc.id for arc in arcs], name="w")
        # model.addConstr(gp.quicksum(w[arc.id] for arc in arcs if arc.tail.id == "root") == 1)
        # model.addConstr(gp.quicksum(w[arc.id] for arc in arcs if arc.head.id == "sink") == 1)
        # model.addConstrs(gp.quicksum(w[arc.id] for arc in u.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in u.incoming_arcs) == 0 for u in nodes if u.id not in ["root", "sink"])
        # model.addConstrs(y[j] == gp.quicksum(w[arc.id] for arc in arcs if arc.player == "follower" and arc.value == 1 and arc.var_index == j) for j in range(Fcols))

        # Strengthening (Fischetti et al, 2017)
        model.addConstrs((y[j] == val for j, val in instance.known_y_values.items()), name="pre-processing")
    
    model._build_time = time() - t0

    model.update()

    return model, time() - t0