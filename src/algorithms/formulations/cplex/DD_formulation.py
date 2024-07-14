from time import time
from docplex.mp.model import Model
import numpy as np

from algorithms.utils.solve_HPR_gurobi import solve as solve_HPR

def get_model(instance, diagram, incumbent):
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

    t0 = time()
    model = Model(log_output=True, float_precision=6)

    x = model.binary_var_dict(Lcols, name="x")
    y = model.binary_var_dict(Fcols, name="y")

    if incumbent:
        warmstart = model.new_solution()
        for j in range(Lcols):
            warmstart.add_var_value(x[j], float(incumbent["x"][j]))
        for j in range(Fcols):
            warmstart.add_var_value(y[j], float(incumbent["y"][j]))
        model.add_mip_start(warmstart)

    model._vars = {
        "x": x,
        "y": y,
    }

    # HPR constrs
    if A.shape[0] != 0 and B.shape[0] != 0:
        model.add_constraints_((model.sum(A[i][j] * x[j] for j in range(Lcols)) + model.sum(B[i][j] * y[j] for j in range(Fcols)) <= a[i] for i in range(Lrows)), names="HPR-leader")
    elif A.shape[0] != 0:
        model.add_constraints_((model.sum(A[i][j] * x[j] for j in range(Lcols)) <= a[i] for i in range(Lrows)), names="HPR-leader")
    elif B.shape[0] != 0:
        model.add_constraints_((model.sum(B[i][j] * y[j] for j in range(Fcols)) <= a[i] for i in range(Lrows)), names="HPR-leader")
    model.add_constraints_((model.sum(C[i][j] * x[j] for j in range(Lcols)) + model.sum(D[i][j] * y[j] for j in range(Fcols)) <= b[i] for i in range(Frows)), names="HPR-follower")

    # Objective function
    obj = model.sum(c_leader[j] * x[j] for j in range(Lcols)) + model.sum(c_follower[j] * y[j] for j in range(Fcols))
    model.minimize(obj)

    if diagram:
        nodes = diagram.nodes
        arcs = diagram.arcs
        root_node = diagram.root_node
        sink_node = diagram.sink_node
        interaction_rows = [i for i in range(instance.Frows) if instance.interaction[i] == "both"]

        pi = model.continuous_var_dict([node.id for node in nodes], lb=-model.infinity, name="pi")
        lamda = model.continuous_var_dict([arc.id for arc in arcs if arc.player == "leader"], name="lambda")
        alpha = model.binary_var_dict([arc.id for arc in arcs if arc.player == "leader"], name="alpha")
        beta = model.binary_var_dict([(arc.id, idx) for arc in arcs for idx in interaction_rows if arc.player == "leader"], name="beta")

        # Value-function constr
        model.add_constraint_(model.sum(d[j] * y[j] for j in range(Fcols)) <= pi[root_node.id], ctname="value-function")

        # Dual feasibility
        model.add_constraints_((pi[arc.tail.id] - pi[arc.head.id] <= arc.cost for arc in arcs if arc.player in ["follower", None]), names="dual-feas-0")
        model.add_constraints_((pi[arc.tail.id] - pi[arc.head.id] - lamda[arc.id] <= 0 for arc in arcs if arc.player == "leader"), names="dual-feas-1")

        # Strong duality
        model.add_constraint_(pi[sink_node.id] == 0, ctname="strong-dual-sink")

        # Gamma bounds
        M = 1e6
        model.add_constraints_((lamda[arc.id] <= (M - arc.tail.follower_cost) * alpha[arc.id] for arc in arcs if arc.player == "leader"), names="gamma-bounds")

        # Alpha-beta relationship
        model.add_constraints_((alpha[arc.id] <= model.sum(beta[arc.id, i] for i in interaction_rows) for arc in arcs if arc.player == "leader"), names="alpha-beta1")
        model.add_constraints_((alpha[arc.id] >= beta[arc.id, i] for i in interaction_rows for arc in arcs if arc.player == "leader"), names="alpha-beta2")

        # Blocking definition
        M = {i: sum(min(C[i][j], 0) for j in range(Lcols)) for i in interaction_rows}
        model.add_constraints_(
            (model.sum(C[i][j] * x[j] for j in range(Lcols)) >= M[i] + beta[arc.id, i] * (-M[i] + instance.b[i] - arc.tail.state[i] + 1) 
            for arc in arcs for i in interaction_rows if arc.player == "leader"), names="blocking-def"
        )

        # Strengthening (Fischetti et al, 2017)
        model.add_constraints_((y[j] == val for j, val in instance.known_y_values.items()), names="pre-processing")

    return model, time() - t0