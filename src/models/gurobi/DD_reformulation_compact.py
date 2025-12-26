from time import time
import gurobipy as gp
import numpy as np

def get_model(instance, diagram, max_follower_value, problem_type, problem_setting, incumbent=None):
    t0 = time()
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
    
    model = gp.Model()

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

    # Strengthening (Fischetti et al, 2017)
    model.addConstrs((y[j] == val for j, val in instance.known_y_values.items()), name="pre-processing")

    # Objective function
    model.setObjective(cL @ x + cF @ y)

    if diagram:
        nodes = diagram.nodes
        arcs = diagram.arcs
        root_node = diagram.root_node
        sink_node = diagram.sink_node
        interaction_rows = [i for i in range(instance.mF) if instance.interaction[i] == "both"]

        pi = model.addVars([node.id for node in nodes], lb=-gp.GRB.INFINITY, name="pi")
        lamda = model.addVars([arc.id for arc in arcs if arc.player == "leader"], name="lambda")
        alpha = model.addVars([arc.id for arc in arcs if arc.player == "leader"], vtype=gp.GRB.BINARY, name="alpha")
        beta = model.addVars(
            [arc.id for arc in arcs if arc.player == "leader"], interaction_rows, 
            vtype=gp.GRB.BINARY, 
            name="beta"
        )

        model._vars.update(
            {
                "pi": pi,
                "lambda": lamda,
                "alpha": alpha,
                "beta": beta
            }
        )

        # Value-function constr
        model.addConstr(d @ y <= pi[root_node.id], name="ValueFunction")

        # Dual feasibility
        model.addConstrs((
            pi[arc.tail.id] - pi[arc.head.id] 
            <= arc.follower_cost for arc in arcs if arc.player == "follower"), 
            name="DualFeas0"
        )
        model.addConstrs((
            pi[arc.tail.id] - pi[arc.head.id] - lamda[arc.id] 
            <= 0 for arc in arcs if arc.player == "leader"), 
            name="DualFeas1"
        )
        model.addConstr(pi[sink_node.id] == 0, name="PiSink")

        # Primal-dual linearization
        model.addConstrs(
            lamda[arc.id] 
            <= (max_follower_value - arc.tail.follower_cost) * alpha[arc.id] for arc in arcs if arc.player == "leader"
        )

        # Alpha-beta relationship
        model.addConstrs(
            alpha[arc.id] 
            <= gp.quicksum(beta[arc.id, i] for i in interaction_rows) for arc in arcs if arc.player == "leader"
        )
        model.addConstrs(
            alpha[arc.id] 
            >= beta[arc.id, i] for i in interaction_rows for arc in arcs if arc.player == "leader"
        )

        # Blocking definition
        M = {i: sum(min(C[i][j], 0) for j in range(nL)) for i in interaction_rows}
        if problem_type == "general":
            model.addConstrs(
                C[i] @ x >= M[i] + beta[arc.id, i] * (-M[i] + instance.b[i] - arc.tail.state[i] + 1) 
                for arc in arcs for i in interaction_rows if arc.player == "leader"
            )
        elif problem_type == "bisp-kc":
            model.addConstrs(
                C[i] @ x 
                >= M[i] + beta[arc.id, i] * (-M[i] + instance.b[i] - arc.tail.state[-len(interaction_rows):][i] + 1) 
                for arc in arcs for i in interaction_rows if arc.player == "leader"
            )
    
    if problem_setting == "pessimistic":
        pi_pess = model.addVars([node.id for node in nodes], lb=-gp.GRB.INFINITY, name="pi_pess")
        lamda_pess = model.addVars([arc.id for arc in arcs if arc.player == "leader"], name="lambda_pess")
        beta_pess = model.addVars([arc.id for arc in arcs if arc.player == "leader"], name="beta_pess")
        delta_pess = model.addVars([arc.id for arc in arcs if arc.player == "leader"], vtype=gp.GRB.BINARY, name="delta_pess")
        
        model._vars.update(
            {
                "pi_pess": pi_pess,
                "lambda_pess": lamda_pess,
                "beta_pess": beta_pess,
                "delta_pess": delta_pess
            }
        )

        # Pessimistic constraints (new DD)
        model.addConstrs((
            pi_pess[arc.tail.id] - pi_pess[arc.head.id] 
            >= arc.leader_cost 
            for arc in arcs if arc.player == "follower"), 
            name="DualFeasPessFollower"
        )
        model.addConstrs((
            pi_pess[arc.tail.id] - pi_pess[arc.head.id] + lamda_pess[arc.id] + beta_pess[arc.id] 
            >= 0 
            for arc in arcs if arc.player == "leader"), 
            name="DualFeasPessLeader"
        )
        model.addConstr(cF @ y >= pi_pess[root_node.id], name="ValueFunctionPess")
        model.addConstr((pi_pess[sink_node.id] == 0), name="PiSinkPess")
        model.addConstrs(
            lamda_pess[arc.id] 
            <= 1e8 * alpha[arc.id] 
            for arc in arcs if arc.player == "leader"
        )

        # Delta definition
        model.addConstrs(
            beta_pess[arc.id] 
            <= 1e8 * delta_pess[arc.id]
            for arc in arcs if arc.player == "leader"
        )
        model.addConstrs(
            arc.tail.state[-2] 
            <= d @ y + 1e8 * delta_pess[arc.id]
            for arc in arcs if arc.player == "leader"
        )
        model.addConstrs(
            arc.tail.state[-2] 
            >= instance.d @ y + 0.5 - 1e8 * (1 - delta_pess[arc.id]) 
            for arc in arcs if arc.player == "leader"
        )

    model._build_time = time() - t0

    model.update()

    return model