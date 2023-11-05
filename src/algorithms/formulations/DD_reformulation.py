import gurobipy as gp
import numpy as np

from algorithms.utils.solve_HPR import run as solve_HPR

def get_model(instance, diagram, time_limit, incumbent):
    Lcols = instance.Lcols
    Lrows = instance.Lrows
    Fcols = instance.Lcols
    Frows = instance.Frows

    A = instance.A
    B = instance.B
    C = instance.C
    D = instance.D
    a = instance.a
    b = instance.b
    c_leader = instance.c_leader
    c_follower = instance.c_follower
    d = instance.d

    nodes = diagram.nodes
    arcs = diagram.arcs
    root_node = nodes["root"]
    sink_node = nodes["sink"]

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    M = solve_HPR(instance, obj="follower", sense="max")[0]

    x = model.addVars(Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(Fcols, vtype=gp.GRB.BINARY, name="y")
    w = model.addVars([arc.id for arc in arcs], ub=1, name="w")
    pi = model.addVars([node.id for node in nodes.values()], lb=-gp.GRB.INFINITY, name="pi")
    lamda = model.addVars([arc.id for arc in arcs if arc.player == "leader" and arc.value == 0], name="lambda")
    beta = model.addVars([arc.id for arc in arcs if arc.player == "leader" and arc.value == 1], name="beta")

    if incumbent:
        for j in range(Lcols):
            x[j].start = incumbent["x"][j]
        for j in range(Fcols):
            y[j].start = incumbent["y"][j]

    vars = {
        "x": x,
        "y": y,
        "w": w,
        "pi": pi,
        "lambda": lamda,
        "beta": beta
    }

    # HPR constrs
    model.addConstrs((gp.quicksum(A[i][j] * x[j] for j in range(Lcols)) + gp.quicksum(B[i][j] * y[j] for j in range(Fcols)) <= a[i] for i in range(Lrows)), name="LeaderHPR")
    model.addConstrs((gp.quicksum(C[i][j] * x[j] for j in range(Lcols)) + gp.quicksum(D[i][j] * y[j] for j in range(Fcols)) <= b[i] for i in range(Frows)), name="FollowerHPR")

    # Value-function constr
    model.addConstr(gp.quicksum(d[j] * y[j] for j in range(Fcols)) <= gp.quicksum(arc.cost * w[arc.id] for arc in arcs), name="ValueFunction")

    # Flow constrs
    model.addConstr(gp.quicksum(w[arc.id] for arc in root_node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in root_node.incoming_arcs) == 1, name="FlowRoot")
    model.addConstr(gp.quicksum(w[arc.id] for arc in sink_node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in sink_node.incoming_arcs) == -1, name="FlowSink")
    model.addConstrs((gp.quicksum(w[arc.id] for arc in node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in node.incoming_arcs) == 0 for node in nodes.values() if node.id not in ["root", "sink"]), name="FlowAll")

    # Arc capacity constrs
    model.addConstrs((w[arc.id] <= x[arc.var_index] for arc in arcs if arc.player == "leader" and arc.value == 1), name="ArcCap1")
    model.addConstrs((w[arc.id] <= 1 - x[arc.var_index] for arc in arcs if arc.player == "leader" and arc.value == 0), name="ArcCap0")

    # Dual feasibility
    model.addConstrs((pi[arc.tail] - pi[arc.head] <= arc.cost for arc in arcs if arc.player in ["follower", None]), name="DualFeasFollower")
    model.addConstrs((pi[arc.tail] - pi[arc.head] - lamda[arc.id] <= 0 for arc in arcs if arc.player == "leader" and arc.value == 0), name="DualFeasLeader0")
    model.addConstrs((pi[arc.tail] - pi[arc.head] - beta[arc.id] <= 0 for arc in arcs if arc.player == "leader" and arc.value == 1), name="DualFeasLeader1")

    # Strong duality
    model.addConstr(pi[root_node.id] == gp.quicksum(arc.cost * w[arc.id] for arc in arcs), name="StrongDualRoot")
    model.addConstr(pi[sink_node.id] == 0, name="StrongDualSink")
    
    # M -= solve_HPR(instance, obj="follower", sense="min")[0]
    # model.addConstrs((lamda[arc.id] <= M * x[arc.var_index] for arc in arcs if arc.player == "leader" and arc.value == 0), name="StrongDual0")
    # model.addConstrs((beta[arc.id] <= M * (1 - x[arc.var_index]) for arc in arcs if arc.player == "leader" and arc.value == 1), name="StrongDual1")

    model.addConstrs((lamda[arc.id] <= (M - nodes[arc.tail].follower_cost) * x[arc.var_index] for arc in arcs if arc.player == "leader" and arc.value == 0), name="StrongDual0")
    model.addConstrs((beta[arc.id] <= (M - nodes[arc.tail].follower_cost) * (1 - x[arc.var_index]) for arc in arcs if arc.player == "leader" and arc.value == 1), name="StrongDual1")

    # Strengthening (Fischetti et al, 2017)
    for j in range(Fcols):
        if np.all([D[i][j] <= 0 for i in range(Frows)]) and d[j] < 0:
            model.addConstr(y[j] == 1)
        elif np.all([D[i][j] >= 0 for i in range(Frows)]) and d[j] > 0:
            model.addConstr(y[j] == 0)

    # Objective function
    obj = gp.quicksum(c_leader[j] * x[j] for j in range(Lcols)) + gp.quicksum(c_follower[j] * y[j] for j in range(Fcols))
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    return model, vars