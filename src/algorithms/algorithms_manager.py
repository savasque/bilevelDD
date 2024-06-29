import logzero
from time import time
import numpy as np
from functools import partial

from classes.decision_diagram import DecisionDiagram

from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

import constants

from .formulations.gurobi.DD_formulation import get_model
# from .formulations.cplex.DD_formulation_compressed_leader import get_model as get_cplex_model_with_compressed_leader

from .gurobi_callback import Callback as GurobiCallback
from .gurobi_callback import CallbackData as GurobiCallbackData
# from .cplex_callback import CplexCallback

from utils.utils import write_modified_model_mps_file, copy_aux_file
from .utils.solve_HPR import solve as solve_HPR
from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem
from .utils.sampler import Sampler


class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def one_time_compilation_approach(self, instance, max_width, ordering_heuristic, discard_method, solver_time_limit, approach):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()

        if max_width > 0:
            # Compile diagram
            diagram = diagram_manager.compile_diagram(
                diagram, instance, ordering_heuristic, 
                method="branching", max_width=max_width, discard_method=discard_method
            )
        else:
            diagram = None

        # Solve reformulation
        result, model = self.solve_DD_reformulation(
            instance, diagram, approach,
            time_limit=solver_time_limit if not diagram else solver_time_limit - diagram.compilation_runtime,
            incumbent=None
        )

        if approach != "write_model":
            # Update final result
            result["approach"] = approach
            result["discard_method"] = discard_method
            result["iters"] = 0

            # Update HPR info
            HPR_value, _, _, HPR_runtime = self.get_HPR_bounds(instance)
            result["HPR"] = HPR_value
            result["HPR_runtime"] = HPR_runtime
            
            if result["bilevel_gap"] < 1e-6:
                result["opt"] = 1
            else:
                result["opt"] = 0

            # Compute continuous relaxation bound
            relaxed_model = model.relax()
            relaxed_model.Params.OutputFlag = 0
            relaxed_model.optimize()
            result["relaxation_obj_val"] = relaxed_model.objval
            
            self.logger.warning("Results for {instance} -> LB: {lower_bound} - UB: {upper_bound} - BilevelGap: {bilevel_gap}% - MIPGap: {mip_gap} - HPR: {HPR} - Runtime: {total_runtime} - DDWidth: {width} - Cuts: {num_cuts}".format(**result))
            self.logger.info(
                "Runtimes -> Compilation: {compilation_runtime} - Ordering heuristic: {ordering_heuristic_runtime} - Model build: {model_build_runtime} - Model solve: {model_runtime} - Cut generation: {cuts_runtime}".format(**result)
            )

        return result

    def iterative_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, discard_method, solver_time_limit):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()

        t0 = time()
        iter = 0
        ys = list()

        # Get HPR bounds
        HPR_value, HPR_optimal_solution, UB, _ = self.get_HPR_bounds(instance)
        self.logger.info("HPR solved -> LB: {} - UB: {}".format(HPR_value, UB))

        # Build set Y
        if constants.SAMPLING_LENGTH:
            Y, sampling_runtime = self.sample_follower_solutions(instance)
            Y = [HPR_optimal_solution["y"]] + Y
        else:
            Y = [HPR_optimal_solution["y"]]
            sampling_runtime = 0

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, discard_method, Y
        )
        initial_width = int(diagram.width)

        # Solve reformulation
        if compilation_method == "follower_then_compressed_leader":
            result, model, vars = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit, use_lazy_cuts=constants.USE_LAZY_CUTS
            )

        self.logger.warning("New bounds -> LB: {lower_bound} - UB: {upper_bound}".format(**result))
        model.Params.OutputFlag = 0

        LB = result["lower_bound"]
        UB = result["upper_bound"]

        # Create new DDs and cuts
        while time() - t0 <= solver_time_limit:
            if instance.d @ result["vars"]["y"] <= instance.d @ result["opt_y"]:
                self.logger.warning("Bilevel solution found!")
                break
            else:
                # Current solution is not bilevel feasible. Add a cut and solve again
                # Build extra DD
                diagram = DecisionDiagram(iter + 1)
                Y = [result["opt_y"]]
                diagram = diagram_manager.compile_diagram(
                    diagram, instance, compilation_method, float("inf"), 
                    ordering_heuristic, discard_method, Y,
                    skip_brute_force_compilation=True
                )
                ys.append(result["opt_y"])
                # Add associated cuts and solve
                self.add_DD_cuts(instance, diagram, model, vars)
                model.Params.TimeLimit = solver_time_limit - (time() - t0)
                self.logger.info("Solving new model with added cuts. Time limit: {} s".format(solver_time_limit - (time() - t0)))
                model.optimize()
                if model.status == 2:
                    self.logger.info("Model succesfully solved -> Time elapsed: {} s".format(model.runtime))
                    result = self.get_results(instance, diagram, model, vars, model_building_runtime=0)
                    if result["upper_bound"] <= UB - 1 or result["lower_bound"] >= LB + 1:
                        LB_diff = max(result["lower_bound"] - LB, 0)
                        UB_diff = max(UB - result["upper_bound"], 0)
                        LB = max(result["lower_bound"], LB)
                        UB = min(result["upper_bound"], UB)
                        self.logger.warning("New bounds -> LB: {} (+{}) - UB: {} (-{})".format(LB, LB_diff, UB, UB_diff))

            iter += 1

        total_runtime = time() - t0

        # Update results
        result["approach"] = "one_time_compilation"
        result["discard_method"] = discard_method
        result["HPR"] = HPR_value
        result["sampling"] = True if len(Y) >= 2 else False
        result["Y_length"] = len(Y)
        result["sampling_runtime"] = sampling_runtime
        result["total_runtime"] = round(total_runtime + instance.load_runtime)
        result["time_limit"] = solver_time_limit
        result["num_nodes"] = diagram.node_count + 2
        result["num_arcs"] = diagram.arc_count
        result["upper_bound"] = min(result["upper_bound"], UB)
        result["bilevel_gap"] = round((result["upper_bound"] - result["lower_bound"]) / abs(result["upper_bound"] + 1e-2), 3) if result["upper_bound"] < float("inf") else None
        result["iters"] = iter
        result["width"] = initial_width
        result["max_width"] = max_width

        self.logger.info("Results for {instance} -> LB: {lower_bound} - UB: {upper_bound} - Iters: {iters} - MIPGap: {mip_gap} - BilevelGap: {bilevel_gap} - HPR: {HPR} - Runtime: {total_runtime} - Iters: {iters}".format(**result))

        return result
    
    def get_HPR_bounds(self, instance):
        UB = float("inf")
        t0 = time()

        self.logger.info("Solving HPR and aux problem")
        HPR_value, HPR_solution = solve_HPR(instance)
        _, y = solve_follower_problem(instance, HPR_solution["x"])  # Optimal follower response
        _, y = solve_aux_problem(instance, HPR_solution["x"], instance.d @ y)  # Break ties among optimal follower responses
        self.logger.info("HPR solved -> Time elpsed: {}".format(time() - t0))
        HPR_optimal_response = {"x": HPR_solution["x"], "y": y}

        if instance.Lrows:
            if np.all([instance.A @ HPR_optimal_response["x"] + instance.B @ HPR_optimal_response["y"] <= instance.a]):
                UB = instance.c_leader @ HPR_optimal_response["x"] + instance.c_follower @ HPR_optimal_response["y"]
        else:
            UB = instance.c_leader @ HPR_optimal_response["x"] + instance.c_follower @ HPR_optimal_response["y"]

        return HPR_value, HPR_optimal_response, UB, time() - t0
    
    def add_DD_cuts(self, instance, diagram, model, vars):
        import gurobipy as gp

        interaction_indices = [i for i in range(instance.Frows) if instance.interaction[i] == "both"]

        # Reset model
        model.reset()

        # Create vars
        w = model.addVars([(arc.id, diagram.id) for arc in diagram.arcs], ub=1, name="w")
        pi = model.addVars([(node.id, diagram.id) for node in diagram.nodes], lb=-gp.GRB.INFINITY, name="pi")
        gamma = model.addVars([(arc.id, diagram.id) for arc in diagram.arcs if arc.player == "leader"], name="gamma")
        alpha = model.addVars([(arc.id, diagram.id) for arc in diagram.arcs if arc.player == "leader"], vtype=gp.GRB.BINARY, name="alpha")
        beta = model.addVars([(arc.id, diagram.id) for arc in diagram.arcs if arc.player == "leader"], interaction_indices, vtype=gp.GRB.BINARY, name="beta")
        
        # Value-function constr
        model.addConstr(gp.quicksum(instance.d[j] * vars["y"][j] for j in range(instance.Fcols)) <= gp.quicksum(arc.cost * w[arc.id, diagram.id] for arc in diagram.arcs), name="ValueFunction")

        # Flow constrs
        model.addConstr(gp.quicksum(w[arc.id, diagram.id] for arc in diagram.root_node.outgoing_arcs) - gp.quicksum(w[arc.id, diagram.id] for arc in diagram.root_node.incoming_arcs) == 1, name="FlowRoot")
        model.addConstr(gp.quicksum(w[arc.id, diagram.id] for arc in diagram.sink_node.outgoing_arcs) - gp.quicksum(w[arc.id, diagram.id] for arc in diagram.sink_node.incoming_arcs) == -1, name="FlowSink")
        model.addConstrs(gp.quicksum(w[arc.id, diagram.id] for arc in node.outgoing_arcs) - gp.quicksum(w[arc.id, diagram.id] for arc in node.incoming_arcs) == 0 for node in diagram.nodes if node.id not in ["root", "sink"])

        # Dual feasibility
        model.addConstrs((pi[arc.tail.id, diagram.id] - pi[arc.head.id, diagram.id] <= arc.cost for arc in diagram.arcs if arc.player in ["follower", None]), name="DualFeasFollower")
        model.addConstrs((pi[arc.tail.id, diagram.id] - pi[arc.head.id, diagram.id] - gamma[arc.id, diagram.id] <= 0 for arc in diagram.arcs if arc.player == "leader"), name="DualFeasLeader")

        # Strong duality
        model.addConstr(pi[diagram.root_node.id, diagram.id] == gp.quicksum(arc.cost * w[arc.id, diagram.id] for arc in diagram.arcs), name="StrongDualRoot")
        model.addConstr(pi[diagram.sink_node.id, diagram.id] == 0, name="StrongDualSink")

        # Gamma bounds
        M_gamma = 1e6
        model.addConstrs(gamma[arc.id, diagram.id] <= M_gamma * alpha[arc.id, diagram.id] for arc in diagram.arcs if arc.player == "leader")

        # Alpha-beta relationship
        model.addConstrs(alpha[arc.id, diagram.id] <= gp.quicksum(beta[arc.id, diagram.id, i] for i in interaction_indices) for arc in diagram.arcs if arc.player == "leader")

        # Blocking definition
        M_blocking = {i: sum(max(-instance.C[i][j], 0) for j in range(instance.Lcols)) for i in range(instance.Frows)}
        model.addConstrs(instance.C[i] @ list(vars["x"].values()) >= -M_blocking[i] + beta[arc.id, diagram.id, i] * (M_blocking[i] + arc.block_values[i]) for arc in diagram.arcs if arc.player == "leader" for i in interaction_indices)     

    def solve_DD_reformulation(self, instance, diagram, approach, time_limit, incumbent=None):
        model, model_building_runtime = get_model(instance, diagram, incumbent)

        if approach == "write_model":
            ######################################################## Write mod model aux file #############################################################
            from utils.utils import write_modified_model_mps_file, copy_aux_file
            write_modified_model_mps_file(model, instance, diagram)
            copy_aux_file(instance, diagram)
            results = {
                "instance": instance.name,
                "nL": instance.Lcols,
                "nF": instance.Fcols,
                "mL": len(instance.a),
                "p": int(instance.name.split("/")[1].split("_")[2][1:]),
                "rhs_ratio": int(instance.name.split("/")[1].split("_")[3][1:]),
                "ddmodel": None,
                "approach": None,
                "max_width": diagram.max_width,
                "ordering_heuristic": None ,
                "lower_bound": None,
                "best_obj_val": None,
                "mip_gap": None,
                "upper_bound": None,
                "bilevel_gap": None,
                "width": 0 if not diagram else diagram.width,
                "total_runtime": None,
                "compilation_runtime": 0 if not diagram else round(diagram.compilation_runtime),
                "ordering_heuristic_runtime": 0 if not diagram else round(diagram.ordering_heuristic_runtime),
                "model_build_runtime": round(model_building_runtime),
                "model_runtime": None,
                "num_vars": model.numVars,
                "num_constrs": model.numConstrs,
                "nodes": None if not diagram else diagram.node_count,
                "arcs": None if not diagram else diagram.arc_count,
                "follower_response": None,
                "SEP": None,
                "solution": None
            }
            ################################################################## * ##########################################################################
        
        elif approach == "relaxation" or approach.split(":")[0] == "lazy_cuts":
            callback_data = GurobiCallbackData(instance)
            callback_func = partial(GurobiCallback().callback, cbdata=callback_data)
            if approach.split(":")[0] == "lazy_cuts":
                model.Params.LazyConstraints = 1
                callback_data.lazy_cuts = True
                callback_data.cuts_type = approach.split(":")[1]
                self.logger.info("Solving DD refomulation with lazy cuts -> Time limit: {} s - Cut types: {} - Sep type: {}".format(time_limit, callback_data.cuts_type, callback_data.bilevel_free_set_sep_type))
            else:
                self.logger.info("Solving DD refomulation relaxation -> Time limit: {} s".format(round(time_limit)))
            
            # Solve DD reformulation
            model.Params.TimeLimit = max(time_limit - model_building_runtime, 0)
            model.optimize(lambda model, where: callback_func(model, where))
            self.logger.info("DD reformulation succesfully solved -> Time elapsed: {} s".format(round(model.runtime)))
            self.logger.debug("LB: {}, MIPGap: {}".format(model.objBound, model.MIPGap))

            results = self.get_gurobi_results(instance, diagram, callback_data, model, model_building_runtime)
        
        else:
            raise ValueError("Invalid approach: {}".format(approach))

        return results, model  

    def get_gurobi_results(self, instance, diagram, callback_data, model, model_building_runtime):
        try:
            objval = model.objVal
        except:
            objval = None
        results = {
            "instance": instance.name,
            "nL": instance.Lcols,
            "nF": instance.Fcols,
            "mL": len(instance.a),
            "mF": len(instance.b),
            "approach": None,
            "max_width": 0 if not diagram else diagram.max_width,
            "ordering_heuristic": None if not diagram else diagram.ordering_heuristic,
            "lower_bound": model.ObjBound,
            "best_obj_val": objval,
            "mip_gap": model.MIPGap,
            "upper_bound": objval,
            "bilevel_gap": 100 * round((objval - model.ObjBound) / abs(objval + 1e-6), 3),
            "width": 0 if not diagram else diagram.width,
            "total_runtime": None,
            "compilation_runtime": 0 if not diagram else round(diagram.compilation_runtime),
            "ordering_heuristic_runtime": 0 if not diagram else round(diagram.ordering_heuristic_runtime),
            "model_build_runtime": round(model_building_runtime),
            "model_runtime": round(model.runtime),
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs,
            "nodes": None if not diagram else diagram.node_count + 2,
            "arcs": None if not diagram else diagram.arc_count,
            "follower_response": None,
            "num_cuts": callback_data.num_cuts,
            "cuts_runtime": round(callback_data.cuts_time),
            "SEP": callback_data.bilevel_free_set_sep_type,
            "solution": None,
        }

        results["total_runtime"] = sum(value for key, value in results.items() if key in ["compilation_runtime", "full_diagram_compilation_runtime", "model_build_runtime", "model_runtime"])

        # Retrieve vars
        if model.status in [2, 9]:
            sol = {
                "x": model._vars["x"].X,
                "y": model._vars["y"].X
            }
            if diagram:
                sol.update({
                    "pi": {key: value.X for key, value in model._vars["pi"].items()},
                    "lambda": [key for key, value in model._vars["lambda"].items() if value.X > 0],
                    "gamma": [key for key, value in model._vars["gamma"].items() if value.X > 0],
                    "alpha": [key for key, value in model._vars["alpha"].items() if value.X > .5],
                    "beta": [key for key, value in model._vars["beta"].items() if value.X > .5]
                })
        else:
            sol = dict()
        results["solution"] = sol

        return results
    
    def get_cplex_results(self, instance, diagram, model, vars, model_building_runtime):
        try:
            objval = model.objective_value
        except:
            objval = None
        results = {
            "approach": None,
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "lower_bound": model.solve_details.best_bound,
            "best_obj_val": objval,
            "mip_gap": model.solve_details.gap,
            "upper_bound": float("inf"),
            "bilevel_gap": None,
            "total_runtime": None,
            "data_load_runtime": round(instance.load_runtime),
            "compilation_runtime": round(diagram.compilation_runtime),
            "reduce_algorithm_runtime": round(diagram.reduce_algorithm_runtime),
            "sampling_runtime": round(diagram.sampling_runtime),
            "model_build_runtime": round(model_building_runtime),
            "model_runtime": round(model.solve_details.time),
            "num_vars": model.solve_details.columns,
            "num_constrs": "TODO",
            "num_nodes": diagram.node_count,
            "num_arcs": diagram.arc_count,
            "num_node_merges": diagram.num_merges
        }
        results["instance"] = instance.name
        results["width"] = diagram.width
        results["initial_width"] = diagram.initial_width
        try:
            vars = {
                "x": [int(i.solution_value + 0.5) for i in vars["x"].values()],
                "y": [int(i.solution_value + 0.5) for i in vars["y"].values()],
                "w": [key for key, value in vars["w"].items() if value.solution_value > 0],
            }
        except:
            vars = dict()
        results["vars"] = vars

        # Compute upper bound
        results["upper_bound"], results["follower_response"] = self.get_upper_bound(instance, vars)

        return results

    def extract_LP_basis(self, model):
        import scipy.sparse as ss

        constr_names_to_indices = {
            c.ConstrName: i for i, c in enumerate(model.getConstrs())
        }
        m = model.NumConstrs
        col_index = 0
        # Initialize the lists to store the row and column indices of non-zero 
        # elements in the basis matrix
        row_indices, col_indices, values, basic_vars = [], [], [], []
        for v in model.getVars():
            col = model.getCol(v)
            
            # Find basic variables
            if v.VBasis == 0:
                basic_vars.append(v)
                for j in range(col.size()):
                    coeff, name = col.getCoeff(j), col.getConstr(j).ConstrName
                    row_index = constr_names_to_indices[name]
                    row_indices.append(row_index)
                    col_indices.append(col_index)
                    values.append(coeff)
                col_index +=1
            
        # Find constraints with slack variable in the basis
        for c in model.getConstrs():
            name = c.ConstrName
            row_index = constr_names_to_indices[name]
            
            if c.CBasis == 0:
                row_indices.append(row_index)
                col_indices.append(col_index)
                values.append(1)
                col_index +=1

        B = ss.csr_matrix((values, (row_indices, col_indices)), shape=(m, m)).A

        return B, basic_vars

    
        import gurobipy as gp

        # SEP-1
        if sep == "SEP-1":
            y_hat = follower_response

            # Build set
            selected_rows = [idx for idx, val in instance.interaction.items() if val != "leader"]
            G_x = np.vstack((instance.C[selected_rows, :], np.zeros(instance.Lcols)))
            G_y = np.vstack((np.zeros((len(selected_rows), instance.Fcols)), -instance.d))
            G = np.hstack((G_x, G_y))
            g_x = instance.b[selected_rows] + 1 - instance.D[selected_rows, :] @ y_hat
            g_y = -instance.d @ y_hat
            g = np.hstack((g_x, g_y))

        # SEP-2
        elif sep == "SEP-2":
            sep_model = gp.Model()
            sep_model.Params.OutputFlag = 0
            selected_rows = [idx for idx, val in instance.interaction.items() if val != "leader"]
            # selected_rows = [i for i in range(instance.Frows)]
            y = sep_model.addVars(instance.Fcols, vtype=gp.GRB.BINARY)
            w = sep_model.addVars(selected_rows, vtype=gp.GRB.BINARY)
            s = sep_model.addVars(selected_rows, lb=-gp.GRB.INFINITY)
            L_max = {i: sum(max(instance.C[i][j], 0) for j in range(instance.Lcols)) for i in selected_rows}
            L_star = {i: instance.C[i] @ x_sol for i in selected_rows}

            sep_model.addConstr(instance.d @ list(y.values()) <= instance.d @ y_sol - 1)
            sep_model.addConstrs(instance.D[i] @ list(y.values()) + s[i] == instance.b[i] for i in selected_rows)
            sep_model.addConstrs(s[i] + (L_max[i] - L_star[i]) * w[i] >= L_max[i] for i in selected_rows)
            sep_model.setObjective(gp.quicksum(w[i] for i in selected_rows), sense=gp.GRB.MINIMIZE)
            sep_model.optimize()

            y_hat = [i.X for i in y.values()]

            # Build set
            selected_rows = [i for i, val in w.items() if val.X > .5]
            G_x = np.vstack((instance.C[selected_rows, :], np.zeros(instance.Lcols)))
            G_y = np.vstack((np.zeros((len(selected_rows), instance.Fcols)), -instance.d))
            G = np.hstack((G_x, G_y))
            g_x = (instance.b + 1 - instance.D @ y_hat)[selected_rows]
            g_y = -instance.d @ y_hat
            g = np.hstack((g_x, g_y))

        # SEP-3
        elif sep == "SEP-3":
            sep_model = gp.Model()
            sep_model.Params.OutputFlag = 0
            selected_rows = [idx for idx, val in instance.interaction.items() if val == "both"]
            delta = sep_model.addVars(instance.Fcols, vtype=gp.GRB.BINARY)
            t = sep_model.addVars(selected_rows)

            sep_model.addConstr(instance.d @ list(delta.values()) <= -1)
            sep_model.addConstrs(instance.D[i] @ list(delta.values()) <= instance.b[i] - instance.C[i] @ x_sol - instance.D[i] @ y_sol for i in selected_rows)
            sep_model.addConstrs(instance.D[i] @ list(delta.values()) <= t[i] for i in selected_rows)
            sep_model.setObjective(gp.quicksum(t[i] for i in selected_rows), sense=gp.GRB.MINIMIZE)
            sep_model.optimize()

            delta_y = [i.X for i in delta.values()]

            # Build set
            G = np.hstack((instance.C, instance.D))[selected_rows, :]
            g = (instance.b + 1 - instance.D @ delta_y)[selected_rows]
            # for j in range(instance.Fcols):
                # e = np.zeros(instance.Fcols)
                # e[j] = -1
                # G = np.vstack((G, np.hstack((np.zeros(instance.Lcols), e))))
                # g = np.append(g, delta_y[j])

                # e = np.zeros(instance.Fcols)
                # e[j] = 1
                # G = np.vstack((G, np.hstack((np.zeros(instance.Lcols), e))))
                # g = np.append(g, 1 - delta_y[j])
        
        return G, g