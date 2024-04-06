import numpy as np
import gurobipy as gp

from constants import CUT_TYPES, BILEVEL_FREE_SET_SEP_TYPE

from .utils.solve_HPR import solve as solve_HPR
from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem


class CallbackData:
    def __init__(self, instance):
        self.instance = instance
        self.root_node_bound = None
        self.was_root_node_visited = False
        self.use_lazy_cuts = False
        self.cut_types = CUT_TYPES
        self.bilevel_free_set_sep_type = BILEVEL_FREE_SET_SEP_TYPE
        self.num_cuts = 0
        for cut_type in self.cut_types:
            if cut_type not in ["no_good_cuts", "informed_no_good_cuts", "ISP_cuts"]:
                raise ValueError("Cut type not found: {}".format(cut_type))

class Callback:
    def callback(self, model, where, cbdata): 
        if where == gp.GRB.Callback.MIPNODE:
            if not cbdata.was_root_node_visited:
                cbdata.root_node_bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)
                cbdata.was_root_node_visited = True
        
        elif where == gp.GRB.Callback.MIPSOL and cbdata.use_lazy_cuts:
            instance = cbdata.instance

            # Retrieve solution
            x = model._vars["x"]
            y = model._vars["y"]
            x_sol = list(model.cbGetSolution(x).values())
            y_sol = list(model.cbGetSolution(y).values())
            follower_value = solve_follower_problem(instance, x_sol)[0]
            follower_response = solve_aux_problem(instance, x_sol, follower_value)[1]

            if instance.d @ y_sol > follower_value + 1e-3:
                # No-good cut
                if "no_good_cuts" in cbdata.cut_types: 
                    M = abs(follower_value)
                    model.cbLazy(instance.d @ list(y.values()) <= follower_value + M * self.hamming_distance(x_sol, x))
                    cbdata.num_cuts += 1

                # Informed no-good cut
                if "informed_no_good_cuts" in CUT_TYPES:
                    G, g = self.build_bilevel_free_set_S(instance, x_sol, y_sol, follower_response)
                    vars_values = x_sol + y_sol
                    vars = list(x.values()) + list(y.values())
                    beta = {i: g[i] - G[i] @ vars_values for i in range(G.shape[0])}
                    G_bar = np.array([[G[i][j] if vars_values[j] == 0 else -G[i][j] for j in range(len(vars_values))] for i in range(G.shape[0])])
                    gamma = {j: max(G_bar[i][j] / beta[i] for i in range(G_bar.shape[0])) for j in range(len(vars_values))}
                    model.cbLazy(gp.quicksum(gamma[j] * vars[j] for j in range(len(vars)) if vars_values[j] == 0) + gp.quicksum(gamma[j] * (1 - vars[j]) for j in range(len(vars)) if vars_values[j] == 1) >= 1)
                    cbdata.num_cuts += 1

            # Update UB
            vars = np.array(list(x.values()) + list(y.values()))
            values = np.array(x_sol + follower_response)
            model.cbSetSolution(vars, values)
    
    def hamming_distance(self, x_0, x):
        return gp.quicksum(x[j] for j in x if x_0[j] == 0) + gp.quicksum(1 - x[j] for j in x if x_0[j] == 1)

    def build_bilevel_free_set_S(self, instance, x_sol, y_sol, follower_response):
        # SEP-1
        if BILEVEL_FREE_SET_SEP_TYPE == "SEP-1":
            y_hat = follower_response

        # SEP-2
        elif BILEVEL_FREE_SET_SEP_TYPE == "SEP-2":
            sep_model = gp.Model()
            sep_model.Params.OutputFlag = 0
            y = sep_model.addVars(instance.Fcols, vtype=gp.GRB.BINARY)
            w = sep_model.addVars(instance.Frows, vtype=gp.GRB.BINARY)
            s = sep_model.addVars(instance.Frows, lb=-gp.GRB.INFINITY)
            L_max = {i: instance.C[i].sum() for i in range(instance.Frows)}
            L_star = {i: instance.C[i] @ x_sol for i in range(instance.Frows)}

            sep_model.addConstr(instance.d @ list(y.values()) <= instance.d @ y_sol - 1)
            sep_model.addConstrs(instance.D[i] @ list(y.values()) + s[i] == instance.b[i] for i in range(instance.Frows))
            sep_model.addConstrs(s[i] + (L_max[i] - L_star[i]) * w[i] >= L_max[i] for i in range(instance.Frows))
            sep_model.setObjective(gp.quicksum(w[i] for i in range(instance.Frows)), sense=gp.GRB.MINIMIZE)
            sep_model.optimize()

            y_hat = [i.X for i in y.values()]

        # Build set
        G_x = np.vstack((instance.C, np.zeros(instance.Lcols)))
        G_y = np.vstack((np.zeros((instance.Frows, instance.Fcols)), -instance.d))
        G = np.hstack((G_x, G_y))
        g_x = instance.b + 1 - instance.D @ y_hat
        g_y = -instance.d @ y_hat
        g = np.hstack((g_x, g_y))
        
        return G, g