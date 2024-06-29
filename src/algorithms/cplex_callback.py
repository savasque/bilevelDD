import numpy as np
import gurobipy as gp

from cplex.callbacks import HeuristicCallback
from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin

from constants import BILEVEL_FREE_SET_SEP_TYPE

from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem

class CplexCallback:
    def __init__(self, instance, model, vars):
        self.num_cuts = 0
        self.instance = instance
        self.model = model
        self.x = vars["x"]
        self.y = vars["y"]
        self.use_lazy_cuts = False
        self.cut_types = None
        self.bilevel_free_set_sep_type = BILEVEL_FREE_SET_SEP_TYPE
    
    def hamming_distance(self, x_0):
        return self.model.sum(self.x[j] for j in self.x if x_0[j] == 0) + self.model.sum(1 - self.x[j] for j in self.x if x_0[j] == 1)
    
    def build_bilevel_free_set_S(self, x_sol, y_sol, follower_response):
        instance = self.instance

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

    def invoke(self, context):
        # Integer solution
        if context.in_candidate():
            model = self.model
            x_sol = [context.get_candidate_point("x_{}".format(j)) for j in range(self.instance.Lcols)]
            y_sol = [context.get_candidate_point("y_{}".format(j)) for j in range(self.instance.Fcols)]
            vars_values = x_sol + y_sol
            vars = list(self.x.values()) + list(self.y.values())
            follower_value = solve_follower_problem(self.instance, x_sol)[0]
            follower_response = solve_aux_problem(self.instance, x_sol, follower_value)[1]

            if self.instance.d @ y_sol > follower_value + 1e-3:
                # No-good cut
                if "no_good_cuts" in self.cut_types: 
                    M = abs(follower_value)
                    lazyct = model.sum(self.instance.d[j] * self.y[j] for j in range(self.instance.Fcols)) <= follower_value + 1e6 * self.hamming_distance(x_sol)
                    lz_lhs, lz_sense, lz_rhs = ModelCallbackMixin.linear_ct_to_cplex(lazyct)
                    context.reject_candidate(constraints=[lz_lhs], senses=lz_sense, rhs=[lz_rhs])
                    self.num_cuts += 1
            
                # Informed no-good cut
                if "informed_no_good_cuts" in self.cut_types:
                    G, g = self.build_bilevel_free_set_S(x_sol, y_sol, follower_response)
                    beta = {i: g[i] - G[i] @ vars_values for i in range(G.shape[0])}
                    G_bar = np.array([[G[i][j] if vars_values[j] == 0 else -G[i][j] for j in range(len(vars_values))] for i in range(G.shape[0])])
                    gamma = {j: max(G_bar[i][j] / beta[i] for i in range(G_bar.shape[0])) for j in range(len(vars_values))}
                    lazyct = model.sum(gamma[j] * vars[j] for j in range(len(vars)) if vars_values[j] == 0) + model.sum(gamma[j] * (1 - vars[j]) for j in range(len(vars)) if vars_values[j] == 1) >= 1
                    lz_lhs, lz_sense, lz_rhs = ModelCallbackMixin.linear_ct_to_cplex(lazyct)
                    context.reject_candidate(constraints=[lz_lhs], senses=lz_sense, rhs=[lz_rhs])
                    self.num_cuts += 1
