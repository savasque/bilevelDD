from time import time
import numpy as np
import gurobipy as gp

from constants import BILEVEL_FREE_SET_SEP_TYPE

from .formulations.gurobi.follower_problem import get_model as get_follower_model
from .formulations.gurobi.aux_problem import get_model as get_aux_model


class CallbackData:
    def __init__(self, instance, full_diagram=None):
        self.instance = instance
        self.full_diagram = full_diagram
        self.follower_model = get_follower_model(instance, [0] * instance.Lcols)
        self.aux_model = get_aux_model(instance, [0] * instance.Lcols, 0)
        self.follower_model.Params.OutputFlag = 0
        self.aux_model.Params.OutputFlag = 0
        self.root_node_bound = None
        self.lazy_cuts = False
        self.was_root_node_visited = False
        self.cuts_type = None
        self.bilevel_free_set_sep_type = BILEVEL_FREE_SET_SEP_TYPE
        self.num_cuts = 0
        self.cuts_time = 0
        self.value_function_pool = dict()

class Callback:
    def callback(self, model, where, cbdata): 
        if where == gp.GRB.Callback.MIPNODE:
            if not cbdata.was_root_node_visited:
                cbdata.root_node_bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)
                cbdata.was_root_node_visited = True
        
        elif where == gp.GRB.Callback.MIPSOL and cbdata.lazy_cuts:
            t0 = time()
            instance = cbdata.instance

            # Retrieve solution
            x = model._vars["x"]
            y = model._vars["y"]
            x_sol = model.cbGetSolution(x).round()
            y_sol = model.cbGetSolution(y).round()

            new_value_function = False

            # Compute value function
            if str(x_sol) in cbdata.value_function_pool:
                follower_value, follower_response = cbdata.value_function_pool[str(x_sol)]
            else:
                new_value_function = True
                self.update_follower_model(instance, cbdata.follower_model, x_sol)
                cbdata.follower_model.optimize()
                follower_value = cbdata.follower_model.ObjVal
                self.update_aux_model(instance, cbdata.aux_model, x_sol, follower_value + 0.5)
                cbdata.aux_model.optimize()
                follower_response = cbdata.aux_model._vars["y"].X.round()
                cbdata.value_function_pool[str(x_sol)] = (follower_value, follower_response)

            # Add cut
            if instance.d @ y_sol > follower_value + 1e-6:
                if new_value_function:
                    if cbdata.cuts_type == "INC+NGC":
                        cbdata.num_cuts += 2
                    else:
                        cbdata.num_cuts += 1
                    cbdata.cuts_time += time() - t0

                # No-good cut
                if cbdata.cuts_type in ["no_good_cuts", "INC+NGC"]: 
                    M = abs(follower_value)
                    model.cbLazy(gp.quicksum(instance.d[j] * y[j].item() for j in range(instance.Fcols)) <= follower_value + M * self.hamming_distance(x_sol, x))

                # Informed no-good cut
                elif cbdata.cuts_type in ["INC", "INC+NGC"]:
                    G, g = self.build_bilevel_free_set_S(instance, x_sol, y_sol, follower_response)
                    vars_values = np.hstack((x_sol, y_sol))
                    vars = [var for var in x] + [var for var in y]
                    beta = {i: g[i] - G[i] @ vars_values for i in range(G.shape[0])}
                    G_bar = np.array([[G[i][j] if vars_values[j] < .5 else -G[i][j] for j in range(len(vars_values))] for i in range(G.shape[0])])
                    gamma = {j: max(G_bar[i][j] / beta[i] for i in range(G_bar.shape[0])) for j in range(len(vars_values))}
                    model.cbLazy(gp.quicksum(gamma[j] * vars[j].item() for j in range(len(vars)) if vars_values[j] < .5) + gp.quicksum(gamma[j] * (1 - vars[j].item()) for j in range(len(vars)) if vars_values[j] > .5) >= 1)

                else:
                    raise ValueError("Non valid type of cut: {}".format(cbdata.cuts_type))
            
            # Update UB
            model.cbSetSolution([var.item() for var in x], x_sol)
            model.cbSetSolution([var.item() for var in y], follower_response)
    
    def hamming_distance(self, x_0, x):
        return gp.quicksum(x[j].item() for j in range(x.size) if x_0[j] < .5) + gp.quicksum(1 - x[j].item() for j in range(x.size) if x_0[j] > .5)

    def update_follower_model(self, instance, model, x):
        model._constrs.RHS = instance.b - instance.C @ x
        model.reset()

    def update_follower_model_DD(self, diagram, model, x):
        for arc in diagram.arcs:
            if arc.value == 1:
                model._constrs[arc].RHS = 1 - x[arc.tail.state[0]]

    def update_aux_model(self, instance, model, x, objval):
        model._constrs["HPR"].RHS = instance.b - instance.C @ x
        model._constrs["objval"].rhs = objval
        model.reset()

    def build_bilevel_free_set_S(self, instance, x_sol, y_sol, follower_response):
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
            sep_model = gp.Model()
            sep_model.Params.OutputFlag = 0
            y = sep_model.addMVar(instance.Fcols, vtype=gp.GRB.BINARY)
            w = sep_model.addMVar(instance.Frows, vtype=gp.GRB.BINARY)
            s = sep_model.addMVar(instance.Frows, lb=-gp.GRB.INFINITY)
            L_max = np.array([sum(max(instance.C[i][j], 0) for j in range(instance.Lcols)) for i in range(instance.Frows)])
            L_star = instance.C @ x_sol

            sep_model.addConstr(instance.d @ y <= instance.d @ y_sol - 1)
            sep_model.addConstr(instance.D @ y + s == instance.b)
            sep_model.addConstrs(s[i] + (L_max[i] - L_star[i]) * w[i] >= L_max[i] for i in range(instance.Frows))
            sep_model.setObjective(np.ones(w.size) @ w, sense=gp.GRB.MINIMIZE)
            sep_model.optimize()

            if sep_model.status == 3:
                # Type 1 separation
                y_hat = follower_response
            else:
                y_hat = y.X

            # Build set
            selected_rows = [i for i, val in instance.interaction.items() if val == "both"] 
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