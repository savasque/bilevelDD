from time import time
from docplex.mp.model import Model
import numpy as np


def get_model(instance, diagram, max_follower_value, incumbent):
    nL = instance.nL
    mL = instance.mL
    nF = instance.nF
    mF = instance.mF
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
    model = Model(log_output=True, float_precision=6)

    x = model.binary_var_dict(nL, name="x")
    y = model.binary_var_dict(nF, name="y")

    if incumbent:
        warmstart = model.new_solution()
        for j in range(nL):
            warmstart.add_var_value(x[j], float(incumbent["x"][j]))
        for j in range(nF):
            warmstart.add_var_value(y[j], float(incumbent["y"][j]))
        model.add_mip_start(warmstart)

    model._vars = {
        "x": x,
        "y": y,
    }

    # HPR constrs
    if mL > 0:
        model.add_constraints_((model.sum(A[i][j] * x[j] for j in range(nL)) + model.sum(B[i][j] * y[j] for j in range(nF)) <= a[i] for i in range(mL)), names="HPR-leader")
    model.add_constraints_((model.sum(C[i][j] * x[j] for j in range(nL)) + model.sum(D[i][j] * y[j] for j in range(nF)) <= b[i] for i in range(mF)), names="HPR-follower")

    # Objective function
    obj = model.sum(cL[j] * x[j] for j in range(nL)) + model.sum(cF[j] * y[j] for j in range(nF))
    model.minimize(obj)

    if diagram:
        nodes = diagram.nodes
        arcs = diagram.arcs
        root_node = diagram.root_node
        sink_node = diagram.sink_node
        interaction_rows = [i for i in range(instance.mF) if instance.interaction[i] == "both"]

        pi = model.continuous_var_dict([node.id for node in nodes], lb=-model.infinity, name="pi")
        lamda = model.continuous_var_dict([arc.id for arc in arcs if arc.player == "leader"], name="lambda")
        alpha = model.binary_var_dict([arc.id for arc in arcs if arc.player == "leader"], name="alpha")
        beta = model.binary_var_dict([(arc.id, idx) for arc in arcs for idx in interaction_rows if arc.player == "leader"], name="beta")

        # Value-function constr
        model.add_constraint_(model.sum(d[j] * y[j] for j in range(nF)) <= pi[root_node.id], ctname="value-function")

        # Dual feasibility
        model.add_constraints_((pi[arc.tail.id] - pi[arc.head.id] <= arc.cost for arc in arcs if arc.player in ["follower", None]), names="dual-feas-0")
        model.add_constraints_((pi[arc.tail.id] - pi[arc.head.id] - lamda[arc.id] <= 0 for arc in arcs if arc.player == "leader"), names="dual-feas-1")

        # Strong duality
        model.add_constraint_(pi[sink_node.id] == 0, ctname="strong-dual-sink")

        # Primal-dual linearization
        model.add_constraints_((lamda[arc.id] <= (max_follower_value - arc.tail.follower_cost) * alpha[arc.id] for arc in arcs if arc.player == "leader"), names="gamma-bounds")

        # Alpha-beta relationship
        model.add_constraints_((alpha[arc.id] <= model.sum(beta[arc.id, i] for i in interaction_rows) for arc in arcs if arc.player == "leader"), names="alpha-beta1")
        model.add_constraints_((alpha[arc.id] >= beta[arc.id, i] for i in interaction_rows for arc in arcs if arc.player == "leader"), names="alpha-beta2")

        # Blocking definition
        M = {i: sum(min(C[i][j], 0) for j in range(nL)) for i in interaction_rows}
        model.add_constraints_(
            (model.sum(C[i][j] * x[j] for j in range(nL)) >= M[i] + beta[arc.id, i] * (-M[i] + instance.b[i] - arc.tail.state[i] + 1) 
            for arc in arcs for i in interaction_rows if arc.player == "leader"), names="blocking-def"
        )

        # Strengthening (Fischetti et al, 2017)
        model.add_constraints_((y[j] == val for j, val in instance.known_y_values.items()), names="pre-processing")
    
    model._build_time = time() - t0

    return model