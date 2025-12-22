from time import time
import numpy as np
from docplex.mp.model import Model

from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin

from constants import BILEVEL_FREE_SET_SEP_TYPE

from models.cplex.follower_problem import get_model as get_follower_model
from models.cplex.aux_problem import get_model as get_aux_model

class CplexCallback:
    def __init__(self, instance, model):
        self.num_cuts = 0
        self.instance = instance
        self.follower_model = get_follower_model(instance)
        self.aux_model = get_aux_model(instance)
        self.model = model
        self.use_lazy_cuts = False
        self.cuts_type = None
        self.bilevel_free_set_sep_type = BILEVEL_FREE_SET_SEP_TYPE
        self.num_cuts = 0
        self.cuts_time = 0
        self.value_function_pool = dict()
     
    def invoke(self, context):
        t0 = time()
        instance = self.instance

        # Incumbent update
        if context.in_candidate():
            model = self.model
            x = self.model._vars["x"]
            y = self.model._vars["y"]
            x_sol = np.array([round(context.get_candidate_point("x_{}".format(j))) for j in range(self.instance.Lcols)])
            y_sol = np.array([round(context.get_candidate_point("y_{}".format(j))) for j in range(self.instance.Fcols)])
            
            new_value_function = False

            # Compute value function
            if str(x_sol) in self.value_function_pool:
                follower_value, follower_response = self.value_function_pool[str(x_sol)]
            else:
                new_value_function = True
                self.update_follower_model(x_sol)
                self.follower_model.solve(clean_before_solve=True)
                follower_value = self.follower_model.objective_value + .5
                self.update_aux_model(x_sol, follower_value)
                self.aux_model.solve(clean_before_solve=True)
                follower_response = np.array([self.aux_model._vars["y"][j].solution_value for j in range(instance.Fcols)])
                self.value_function_pool[str(x_sol)] = (follower_value, follower_response)

            # Add cut
            if self.instance.d @ y_sol > follower_value + 1e-3:
                if new_value_function:
                    if self.cuts_type == "INC+NGC":
                        self.num_cuts += 2
                    else:
                        self.num_cuts += 1
                    self.cuts_time += time() - t0

                # No-good cut
                if self.cuts_type in ["no_good_cuts", "INC+NGC"]:
                    M = abs(follower_value)
                    lazyct = model.sum(self.instance.d[j] * y[j] for j in range(self.instance.Fcols)) <= follower_value + M * self.hamming_distance(x_sol, x)
                    lz_lhs, lz_sense, lz_rhs = ModelCallbackMixin.linear_ct_to_cplex(lazyct)
                    context.reject_candidate(constraints=[lz_lhs], senses=lz_sense, rhs=[lz_rhs])
            
                # Informed no-good cut
                elif self.cuts_type in ["INC", "INC+NGC"]:
                    G, g = self.build_bilevel_free_set_S(x_sol, y_sol, follower_response)
                    vars_values = np.hstack((x_sol, y_sol))
                    vars = [var for var in x] + [var for var in y]
                    beta = {i: g[i] - G[i] @ vars_values for i in range(G.shape[0])}
                    G_bar = np.array([[G[i][j] if vars_values[j] < .5 else -G[i][j] for j in range(len(vars_values))] for i in range(G.shape[0])])
                    gamma = {j: max(G_bar[i][j] / beta[i] for i in range(G_bar.shape[0])) for j in range(len(vars_values))}
                    lazyct = model.sum(gamma[j] * vars[j] for j in range(len(vars)) if vars_values[j] < .5) + model.sum(gamma[j] * (1 - vars[j]) for j in range(len(vars)) if vars_values[j] > .5) >= 1
                    lz_lhs, lz_sense, lz_rhs = ModelCallbackMixin.linear_ct_to_cplex(lazyct)
                    context.reject_candidate(constraints=[lz_lhs], senses=lz_sense, rhs=[lz_rhs])

                else:
                    raise ValueError("Non valid type of cut: {}".format(self.cuts_type))

    def hamming_distance(self, x_0, x):
        return self.model.sum(x[j] for j in range(self.instance.Lcols) if x_0[j] < .5) + self.model.sum(1 - x[j] for j in range(self.instance.Lcols) if x_0[j] > .5)
    
    def update_follower_model(self, x):
        instance = self.instance
        for i in range(instance.Frows):
            self.follower_model._constrs[i].rhs = float(instance.b[i] - instance.C[i] @ x)
    
    def update_aux_model(self, x, objval):
        instance = self.instance
        for i in range(instance.Frows):
            self.aux_model._constrs["HPR"][i].rhs = float(instance.b[i] - instance.C[i] @ x)
        self.aux_model._constrs["objval"].rhs = objval

    def build_bilevel_free_set_S(self, x_sol, y_sol, follower_response):
        instance = self.instance

        # SEP-1
        if BILEVEL_FREE_SET_SEP_TYPE == "SEP-1":
            y_hat = follower_response

            # Build set
            selected_rows = [i for i, val in instance.interaction.items() if val == "both"] 
            G_x = np.vstack((instance.C[selected_rows, :], np.zeros(instance.Lcols)))
            G_y = np.vstack((np.zeros((len(selected_rows), instance.Fcols)), -instance.d))
            G = np.hstack((G_x, G_y))
            g_x = (instance.b + 1 - instance.D @ y_hat)[selected_rows]
            g_y = -instance.d @ y_hat
            g = np.hstack((g_x, g_y))

        # SEP-2
        elif BILEVEL_FREE_SET_SEP_TYPE == "SEP-2":
            sep_model = Model(log_output=False, float_precision=6)
            y = sep_model.binary_var_dict(instance.Fcols)
            w = sep_model.binary_var_dict(instance.Frows)
            s = sep_model.continuous_var_dict(instance.Frows, lb=-sep_model.infinity)
            L_max = np.array([sum(max(instance.C[i][j], 0) for j in range(instance.Lcols)) for i in range(instance.Frows)])
            L_star = instance.C @ x_sol

            sep_model.add_constraint_(sep_model.sum(instance.d[j] * y[j] for j in range(instance.Fcols)) <= instance.d @ y_sol - 1)
            sep_model.add_constraints_(sep_model.sum(instance.D[i][j] * y[j] for j in range(instance.Fcols)) + s[i] == instance.b[i] for i in range(instance.Frows))
            sep_model.add_constraints_(s[i] + (L_max[i] - L_star[i]) * w[i] >= L_max[i] for i in range(instance.Frows))
            obj = sep_model.sum(w[i] for i in range(instance.Frows))
            sep_model.minimize(obj)
            
            sep_model.solve()

            y_hat = np.array([round(i.solution_value) for i in y.values()])

            # Build set
            selected_rows = [i for i, val in instance.interaction.items() if val == "both" and w[i].solution_value > .5] 
            G_x = np.vstack((instance.C[selected_rows, :], np.zeros(instance.Lcols)))
            G_y = np.vstack((np.zeros((len(selected_rows), instance.Fcols)), -instance.d))
            G = np.hstack((G_x, G_y))
            g_x = (instance.b + 1 - instance.D @ y_hat)[selected_rows]
            g_y = -instance.d @ y_hat
            g = np.hstack((g_x, g_y))
        
        # # SEP-3
        # elif BILEVEL_FREE_SET_SEP_TYPE == "SEP-3":
        #     sep_model = gp.Model()
        #     sep_model.Params.OutputFlag = 0
        #     delta = sep_model.addVars(instance.Fcols, vtype=gp.GRB.BINARY)
        #     t = sep_model.addVars(instance.Frows)

        #     sep_model.addConstr(instance.d @ list(delta.values()) <= -1)
        #     sep_model.addConstrs(instance.D[i] @ list(delta.values()) <= instance.b[i] - instance.C[i] @ x_sol - instance.D[i] @ y_sol for i in range(instance.Frows))
        #     sep_model.addConstrs(instance.D[i] @ list(delta.values()) <= t[i] for i in range(instance.Frows))
        #     sep_model.setObjective(gp.quicksum(t[i] for i in range(instance.Frows)), sense=gp.GRB.MINIMIZE)
        #     sep_model.optimize()

        #     delta_y = [i.X for i in delta.values()]

        #     # Build set
        #     G = np.hstack((instance.C, instance.D))
        #     g = (instance.b + 1 - instance.D @ delta_y)
        #     # for j in range(instance.Fcols):
        #         # e = np.zeros(instance.Fcols)
        #         # e[j] = -1
        #         # G = np.vstack((G, np.hstack((np.zeros(instance.Lcols), e))))
        #         # g = np.append(g, delta_y[j])

        #         # e = np.zeros(instance.Fcols)
        #         # e[j] = 1
        #         # G = np.vstack((G, np.hstack((np.zeros(instance.Lcols), e))))
        #         # g = np.append(g, 1 - delta_y[j])
        
        return G, g