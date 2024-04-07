from time import time
import gurobipy as gp
import numpy as np

from algorithms.utils.solve_HPR import solve as solve_HPR

def get_model(instance, diagram, time_limit, incumbent=None):
    Lcols = instance.Lcols
    Lrows = instance.Lrows
    Fcols = instance.Fcols
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
    root_node = diagram.root_node
    sink_node = diagram.sink_node
    interaction_indices = [i for i in range(instance.Frows) if np.any(instance.C[i]) and np.any(instance.D[i])]

    t0 = time()
    model = gp.Model()
    model.Params.TimeLimit = time_limit
    # model.Params.IntegralityFocus = 1

    x = model.addVars(Lcols, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(Fcols, vtype=gp.GRB.BINARY, name="y")
    w = model.addVars([arc.id for arc in arcs], ub=1, name="w")
    pi = model.addVars([node.id for node in nodes], lb=-gp.GRB.INFINITY, name="pi")
    gamma = model.addVars([arc.id for arc in arcs if arc.player == "leader"], name="gamma")
    alpha = model.addVars([arc.id for arc in arcs if arc.player == "leader"], vtype=gp.GRB.BINARY, name="alpha")
    beta = model.addVars([arc.id for arc in arcs if arc.player == "leader"], interaction_indices, vtype=gp.GRB.BINARY, name="beta")

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
        "gamma": gamma,
        "alpha": alpha,
        'beta': beta
    }

    model._vars = vars

    # HPR constrs
    model.addConstrs((A[i] @ list(x.values()) + B[i] @ list(y.values()) <= a[i] for i in range(Lrows)), name="LeaderHPR")
    model.addConstrs((C[i] @ list(x.values()) + D[i] @ list(y.values()) <= b[i] for i in range(Frows)), name="FollowerHPR")

    # Value-function constr
    model.addConstr(gp.quicksum(d[j] * y[j] for j in range(Fcols)) <= gp.quicksum(arc.cost * w[arc.id] for arc in arcs), name="ValueFunction")

    # Flow constrs
    model.addConstr(gp.quicksum(w[arc.id] for arc in root_node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in root_node.incoming_arcs) == 1, name="FlowRoot")
    model.addConstr(gp.quicksum(w[arc.id] for arc in sink_node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in sink_node.incoming_arcs) == -1, name="FlowSink")
    model.addConstrs(gp.quicksum(w[arc.id] for arc in node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in node.incoming_arcs) == 0 for node in nodes if node.id not in ["root", "sink"])

    # Dual feasibility
    model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] <= arc.cost for arc in arcs if arc.player in ["follower", None]), name="DualFeasFollower")
    model.addConstrs((pi[arc.tail.id] - pi[arc.head.id] - gamma[arc.id] <= 0 for arc in arcs if arc.player == "leader"), name="DualFeasLeader")

    # Strong duality
    model.addConstr(pi[root_node.id] == gp.quicksum(arc.cost * w[arc.id] for arc in arcs), name="StrongDualRoot")
    model.addConstr(pi[sink_node.id] == 0, name="StrongDualSink")

    # Gamma bounds
    M = 1e6 #solve_HPR(instance, obj="follower", sense="maximize")[0] - solve_HPR(instance, obj="follower", sense="minimize")[0]
    model.addConstrs(gamma[arc.id] <= (M - arc.tail.follower_cost) * alpha[arc.id] for arc in arcs if arc.player == "leader")

    # Alpha-beta relationship
    model.addConstrs(alpha[arc.id] <= gp.quicksum(beta[arc.id, i] for i in interaction_indices) for arc in arcs if arc.player == "leader")

    # Blocking definition
    M_blocking = {i: -sum(min(C[i][j], 0) for j in range(Lcols)) for i in range(Frows)}
    model.addConstrs(
        C[i] @ list(x.values()) >= -M_blocking[i] + beta[arc.id, i] * (M_blocking[i] + arc.block_values[i]) 
        for arc in arcs if arc.player == "leader" for i in interaction_indices
    )

    # Strengthening (Fischetti et al, 2017)
    model.addConstrs(y[j] == val for j, val in instance.known_y_values.items())

    # Objective function
    obj = c_leader @ list(x.values()) + c_follower @ list(y.values())
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    return model, vars, time() - t0