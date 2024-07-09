import logzero
from time import time
import numpy as np
from functools import partial

import gurobipy as gp

from classes.decision_diagram import DecisionDiagram

from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from .formulations.gurobi.DD_formulation import get_model as get_gurobi_model
from .formulations.gurobi.follower_problem import get_model as get_gurobi_follower_model
from .formulations.gurobi.aux_problem import get_model as get_gurobi_aux_model
from .formulations.cplex.DD_formulation import get_model as get_cplex_model
from .formulations.cplex.follower_problem import get_model as get_cplex_follower_model
from .formulations.cplex.aux_problem import get_model as get_cplex_aux_model

from .gurobi_callback import Callback as GurobiCallback
from .gurobi_callback import CallbackData as GurobiCallbackData

from .utils.solve_HPR import solve as solve_HPR


class AlgorithmsManager:
    def __init__(self, instance, num_threads, solver):
        self.logger = logzero.logger
        self.instance = instance
        self.solver = solver
        if solver == "gurobi":
            self.follower_model = get_gurobi_follower_model(instance, [0] * instance.Lcols)
            self.aux_model = get_gurobi_aux_model(instance, [0] * instance.Lcols, 0)
            self.follower_model.Params.OutputFlag = 0
            self.aux_model.Params.OutputFlag = 0
            self.follower_model.Params.Threads = num_threads
            self.aux_model.Params.Threads = num_threads
        elif solver == "cplex":
            self.follower_model = get_cplex_follower_model(instance, [0] * instance.Lcols)
            self.aux_model = get_cplex_aux_model(instance, [0] * instance.Lcols, 0)
            self.follower_model.Params.OutputFlag = 0
            self.aux_model.Params.OutputFlag = 0
            self.follower_model.Params.Threads = num_threads
            self.aux_model.Params.Threads = num_threads
        self.num_threads = num_threads

    def one_time_compilation_approach(self, instance, max_width, ordering_heuristic, discard_method, solver_time_limit, approach):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()

        self.logger.warning("Initializing solver -> Time limit: {}".format(solver_time_limit))

        if max_width > 0:
            # Compile diagram
            diagram = diagram_manager.compile_diagram(
                diagram, instance, ordering_heuristic, 
                method="branching", max_width=max_width, discard_method=discard_method
            )
        else:
            diagram = None

        # Get HPR info
        HPR_value, _, follower_value, follower_response, _, _, HPR_runtime, follower_response_runtime = self.get_HPR_bounds(instance)

        # Solve reformulation
        result, model = self.solve_DD_reformulation(
            instance, diagram, approach,
            time_limit=solver_time_limit if not diagram else solver_time_limit - diagram.compilation_runtime,
            HPR_info={"follower_value": follower_value, "follower_response": follower_response}
        )

        if approach != "write_model":
            # Include HPR info
            result["HPR"] = HPR_value
            result["HPR_runtime"] = HPR_runtime

            # Update final result
            result["approach"] = approach
            result["discard_method"] = discard_method
            result["cuts_runtime"] += round(follower_response_runtime)
            result["iters"] = 0
            if result["bilevel_gap"] < 1e-6:
                result["opt"] = 1
            else:
                result["opt"] = 0

            # Compute continuous relaxation bound
            relaxed_model = model.relax()
            relaxed_model.Params.OutputFlag = 0
            relaxed_model.optimize()
            result["relaxation_obj_val"] = relaxed_model.objval
            
            self.logger.warning(
                "Results for {instance} -> LB: {lower_bound} - UB: {upper_bound} - BilevelGap: {bilevel_gap}% - MIPGap: {mip_gap} - HPR: {HPR} - Runtime: {total_runtime} - DDWidth: {width} - Cuts: {num_cuts}".format(**result)
            )
            self.logger.info(
                "Runtimes -> Compilation: {compilation_runtime} - Ordering heuristic: {ordering_heuristic_runtime} - Model build: {model_build_runtime} - Model solve: {model_runtime} - Cut generation: {cuts_runtime}".format(**result)
            )

        return result

    def iterative_approach(self, instance, max_width, ordering_heuristic, discard_method, solver_time_limit):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()

        self.logger.warning("Initializing solver -> Time limit: {}".format(solver_time_limit))

        t0 = time()
        iter = 0

        if max_width > 0:
            # Compile diagram
            diagram = diagram_manager.compile_diagram(
                diagram, instance, ordering_heuristic, 
                method="branching", max_width=max_width, discard_method=discard_method
            )
        else:
            diagram = None

        # Get HPR info
        HPR_value, _, follower_value, follower_response, UB, bilevel_gap, HPR_runtime, cuts_time = self.get_HPR_bounds(instance)

        # Solve DD relaxation
        result, model = self.solve_DD_reformulation(
            instance, diagram, "relaxation",
            time_limit=solver_time_limit if not diagram else solver_time_limit - diagram.compilation_runtime,
            HPR_info={"follower_value": follower_value, "follower_response": follower_response}
        )
        model_time = result["model_runtime"]
        model_build_time = result["model_build_runtime"]

        # Update bounds
        cuts_time += self.update_upper_bound(instance, result)
        UB = result["upper_bound"]
        LB = result["lower_bound"]

        self.logger.info("Initial bounds -> LB: {lower_bound} - UB: {upper_bound} - Bilevel gap: {bilevel_gap}%".format(**result))

        model.Params.OutputFlag = 0
        best_result = result

        # Main iteration
        while time() - t0 <= solver_time_limit:
            # Stopping criteria
            if result["bilevel_gap"] <= 1e-6:
                self.logger.warning("Bilevel solution found!")
                break

            # Refine model
            else:
                # Current solution is not bilevel feasible. Update bounds, add a cut and solve again
                model_build_time += self.add_DD_cuts(instance, model, result)
                if self.solver == "gurobi":
                    model.Params.TimeLimit = max(solver_time_limit - (time() - t0), 0)
                elif self.solver == "cplex":
                    model.Params.TimeLimit = max(solver_time_limit - (time() - t0), 0)
                
                self.logger.debug("Solving new model with added cuts. Updated time limit: {} s".format(round(solver_time_limit - (time() - t0))))

                if self.solver == "gurobi":
                    model.optimize()
                    model_time += model.runtime
                elif self.solver == "cplex":
                    model.optimize()
                    model_time += model.runtime

                # Solved model
                if model.status == 2:
                    self.logger.debug("Iter {} -> Model succesfully solved -> Time elapsed: {} s".format(iter + 1, round(model.runtime)))
                    result = self.get_results(instance, diagram, model, 0)
                
                    # Update bounds
                    cuts_time += self.update_upper_bound(instance, result)

                    LB_diff = max(result["lower_bound"] - LB, 0)
                    UB_diff = min(result["upper_bound"] - UB, 0)
                    LB = max(LB, result["lower_bound"])
                    UB = min(UB, result["upper_bound"])
                    bilevel_gap = 100 * round((UB - LB) / abs(UB + 1e-6), 6)

                    # Update bounds
                    if LB_diff > .5 or UB_diff < -.5:
                        self.logger.info("New bounds -> LB: {} (+{}) - UB: {} ({}) - Bilevel gap: {}%".format(LB, LB_diff, UB, UB_diff, bilevel_gap))
                        best_result = result

                iter += 1

        # Update final result
        best_result["approach"] = "iterative"
        best_result["discard_method"] = discard_method
        best_result["model_build_runtime"] = round(model_build_time)
        best_result["model_runtime"] = round(model_time)
        best_result["cuts_runtime"] = round(cuts_time)
        best_result["total_runtime"] = round(time() - t0)
        best_result["iters"] = iter
        if best_result["bilevel_gap"] < 1e-6:
            best_result["opt"] = 1
        else:
            best_result["opt"] = 0

        # Include HPR info
        best_result["HPR"] = HPR_value
        best_result["HPR_runtime"] = round(HPR_runtime)

        # Compute continuous relaxation bound
        relaxed_model = model.relax()
        relaxed_model.Params.OutputFlag = 0
        relaxed_model.optimize()
        best_result["relaxation_obj_val"] = relaxed_model.objval

        self.logger.warning(
            "Results for {instance} -> LB: {lower_bound} - UB: {upper_bound} - BilevelGap: {bilevel_gap}% - MIPGap: {mip_gap} - HPR: {HPR} - Runtime: {total_runtime} - DDWidth: {width} - Iters: {iters}".format(**best_result)
        )
        self.logger.info(
            "Runtimes -> Compilation: {compilation_runtime} - Ordering heuristic: {ordering_heuristic_runtime} - Model build: {model_build_runtime} - Model solve: {model_runtime} - Cut generation: {cuts_runtime}".format(**best_result)
        )

        return best_result
    
    def solve_DD_reformulation(self, instance, diagram, approach, time_limit, HPR_info):
        solver = self.solver
        if solver == "gurobi":
            model, model_building_runtime = get_gurobi_model(instance, diagram)
        elif solver == "cplex":
            model, model_building_runtime = get_cplex_model(instance, diagram)

        # Add cut associated to HPR
        model_building_runtime += self.add_DD_cuts(instance, model, {"follower_value": HPR_info["follower_value"], "follower_response": HPR_info["follower_response"]})

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
            if solver == "gurobi":
                callback_data = GurobiCallbackData(instance)
                callback_func = partial(GurobiCallback().callback, cbdata=callback_data)
                model._cbdata = callback_data
                if approach.split(":")[0] == "lazy_cuts":
                    model.Params.LazyConstraints = 1
                    callback_data.lazy_cuts = True
                    callback_data.cuts_type = approach.split(":")[1]
                    self.logger.info("Solving DD refomulation with lazy cuts -> Solver: {} - Time limit: {} s - Cut types: {} - Sep type: {}".format(
                        self.solver, time_limit, callback_data.cuts_type, callback_data.bilevel_free_set_sep_type
                    ))
                else:
                    self.logger.info("Solving DD refomulation relaxation -> Solver: {} - Time limit: {} s".format(self.solver, round(time_limit)))
                
                # Solve DD reformulation
                model.Params.TimeLimit = max(time_limit - model_building_runtime, 0)
                model.Params.Threads = self.num_threads
                model.Params.NumericFocus = 1
                model.optimize(lambda model, where: callback_func(model, where))
                self.logger.info("DD reformulation succesfully solved -> LB: {} - MIPGap: {} - Time elapsed: {} s".format(
                    model.objBound if model.MIPGap > 1e-6 else model.ObjVal, model.MIPGap, round(model.runtime)
                ))

                results = self.get_results(instance, diagram, model, model_building_runtime)
            
            elif solver == "cplex":
                callback_data = GurobiCallbackData(instance)
                callback_func = partial(GurobiCallback().callback, cbdata=callback_data)
                model._cbdata = callback_data
                if approach.split(":")[0] == "lazy_cuts":
                    model.Params.LazyConstraints = 1
                    callback_data.lazy_cuts = True
                    callback_data.cuts_type = approach.split(":")[1]
                    self.logger.info("Solving DD refomulation with lazy cuts -> Time limit: {} s - Cut types: {} - Sep type: {}".format(time_limit, callback_data.cuts_type, callback_data.bilevel_free_set_sep_type))
                else:
                    self.logger.info("Solving DD refomulation relaxation -> Time limit: {} s".format(round(time_limit)))
                
                # Solve DD reformulation
                model.Params.TimeLimit = max(time_limit - model_building_runtime, 0)
                model.Params.Threads = self.num_threads
                model.Params.NumericFocus = 1
                model.optimize(lambda model, where: callback_func(model, where))
                self.logger.info("DD reformulation succesfully solved -> LB: {} - MIPGap: {} - Time elapsed: {} s".format(
                    model.objBound if model.MIPGap > 1e-6 else model.ObjVal, model.MIPGap, round(model.runtime)
                ))

                results = self.get_gurobi_results(instance, diagram, model, model_building_runtime)
        
        else:
            raise ValueError("Invalid approach: {}".format(approach))
            
        return results, model

    def get_follower_response(self, instance, x):
        t0 = time()

        follower_model = self.follower_model
        aux_model = self.aux_model

        # Solve models
        if self.solver == "gurobi":
            self.update_follower_model(instance, x)
            follower_model.optimize()
            follower_value = follower_model.ObjVal + .5
            self.update_aux_model(instance, x, follower_value)
            aux_model.optimize()
            follower_response = aux_model._vars["y"].X
        elif self.solver == "cplex":
            self.update_follower_model(instance, x)
            follower_model.optimize()
            follower_value = follower_model.ObjVal + .5
            self.update_aux_model(instance, x, follower_value)
            aux_model.optimize()
            follower_response = aux_model._vars["y"].X

        return follower_value, follower_response, time() - t0

    def update_upper_bound(self, instance, result):
        result["upper_bound"] = float("inf")
        result["bilevel_gap"] = float("inf")

        # Solve follower problem
        result["follower_value"], result["follower_response"], runtime = self.get_follower_response(instance, result["solution"]["x"])

        # Check follower feasibility for leader problem
        if self.check_leader_feasibility(instance, result["solution"]["x"], result["follower_response"]):
            result["upper_bound"] = instance.c_leader @ result["solution"]["x"] + instance.c_follower @ result["follower_response"]
            result["bilevel_gap"] = 100 * round((result["upper_bound"] - result["lower_bound"]) / abs(result["upper_bound"] + 1e-6), 6)

        return runtime

    def get_HPR_bounds(self, instance):
        UB = float("inf")
        bilevel_gap = float("inf")

        self.logger.debug("Computing HPR bounds")
        
        # Solve HPR
        t0 = time()
        HPR_value, HPR_solution = solve_HPR(instance)
        HPR_runtime = time() - t0

        # Solve subproblems
        follower_value, follower_response, runtime = self.get_follower_response(instance, HPR_solution["x"])

        self.logger.debug("HPR solved -> Time elpsed: {}".format(round(time() - t0)))

        HPR_solution = {"x": HPR_solution["x"], "y": HPR_solution["y"]}

        # Check follower feasibility for leader problem
        if self.check_leader_feasibility(instance, HPR_solution["x"], follower_response):
            UB = instance.c_leader @ HPR_solution["x"] + instance.c_follower @ follower_response
            bilevel_gap = 100 * round((UB - HPR_value) / abs(UB + 1e-6), 6)

        return HPR_value, HPR_solution, follower_value, follower_response, UB, bilevel_gap, HPR_runtime, runtime
    
    def check_leader_feasibility(self, instance, x, y):
        if instance.Lrows == 0:
            return True
        
        if np.all([instance.A @ x + instance.B @ y <= instance.a]):
            return True
        
        return False
    
    def add_DD_cuts(self, instance, model, result):
        t0 = time()
        interaction_rows = [i for i in range(instance.Frows) if instance.interaction[i] == "both"]

        if self.solver == "gurobi":
            # Reset model
            model.reset()

            # Create vars
            alpha = model.addVar(vtype=gp.GRB.BINARY)
            beta = model.addVars(interaction_rows, vtype=gp.GRB.BINARY)
            
            # Alpha-beta relationship
            model.addConstrs(alpha <= 1 - beta[i] for i in interaction_rows)
            model.addConstr(alpha >= gp.quicksum(1 - beta[i] for i in interaction_rows) - (len(interaction_rows) - 1))

            # Blocking definition
            M = {i: sum(min(instance.C[i][j], 0) for j in range(instance.Lcols)) for i in interaction_rows}
            model.addConstrs(
                instance.C[i] @ model._vars["x"] >= M[i] + beta[i] * (-M[i] + instance.b[i] - instance.D[i] @ result["follower_response"] + 1) 
                for i in interaction_rows
            )  

            # Value-function bound
            M = 1e6
            model.addConstr(instance.d @ model._vars["y"] <= result["follower_value"] + M * (1 - alpha))
        
        elif self.solver == "cplex":
            # Reset model
            model.reset()

            # Create vars
            alpha = model.addVar(vtype=gp.GRB.BINARY)
            beta = model.addVars(interaction_rows, vtype=gp.GRB.BINARY)
            
            # Alpha-beta relationship
            model.addConstrs(alpha <= 1 - beta[i] for i in interaction_rows)
            model.addConstr(alpha >= gp.quicksum(1 - beta[i] for i in interaction_rows) - (len(interaction_rows) - 1))

            # Blocking definition
            M = {i: sum(min(instance.C[i][j], 0) for j in range(instance.Lcols)) for i in interaction_rows}
            model.addConstrs(
                instance.C[i] @ model._vars["x"] >= M[i] + beta[i] * (-M[i] + instance.b[i] - instance.D[i] @ result["follower_response"] + 1) 
                for i in interaction_rows
            )  

            # Value-function bound
            M = 1e6
            model.addConstr(instance.d @ model._vars["y"] <= result["follower_value"] + M * (1 - alpha))

        return time() - t0  
    
    def update_follower_model(self, instance, x):
        if self.solver == "gurobi":
            self.follower_model._constrs.RHS = instance.b - instance.C @ x
            self.follower_model.reset()
        elif self.solver == "cplex":
            self.follower_model._constrs.RHS = instance.b - instance.C @ x
            self.follower_model.reset()
    
    def update_aux_model(self, instance, x, objval):
        if self.solver == "gurobi":
            self.aux_model._constrs["HPR"].RHS = instance.b - instance.C @ x
            self.aux_model._constrs["objval"].RHS = objval
            self.aux_model.reset()
        elif self.solver == "cplex":
            self.aux_model._constrs["HPR"].RHS = instance.b - instance.C @ x
            self.aux_model._constrs["objval"].RHS = objval
            self.aux_model.reset()

    def get_results(self, instance, diagram, model, model_building_runtime):
        ## Gurobi results
        if self.solver == "gurobi":
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
                "num_threads": self.num_threads,
                "solver": "gurobi",
                "approach": None,
                "max_width": 0 if not diagram else diagram.max_width,
                "ordering_heuristic": None if not diagram else diagram.ordering_heuristic,
                "lower_bound": model.ObjBound,
                "best_obj_val": objval,
                "mip_gap": model.MIPGap,
                "upper_bound": objval,
                "bilevel_gap": 100 * round((objval - model.ObjBound) / abs(objval + 1e-6), 6),
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
                "num_cuts": model._cbdata.num_cuts,
                "cuts_runtime": round(model._cbdata.cuts_time),
                "SEP": model._cbdata.bilevel_free_set_sep_type,
                "solution": None
            }

            results["total_runtime"] = sum(value for key, value in results.items() if key in ["compilation_runtime", "model_build_runtime", "model_runtime"])

            # Retrieve vars
            try:
                sol = {
                    "x": model._vars["x"].X,
                    "y": model._vars["y"].X
                }
                if diagram:
                    sol.update({
                        "pi": {key: value.X for key, value in model._vars["pi"].items()},
                        "lambda": [key for key, value in model._vars["lambda"].items() if value.X > 0],
                        "alpha": [key for key, value in model._vars["alpha"].items() if value.X > .5],
                        "beta": [key for key, value in model._vars["beta"].items() if value.X > .5]
                    })
            except:
                sol = dict()
            results["solution"] = sol

        ## CPLEX results
        elif self.solver == "cplex":
            try:
                objval = model.objective_value
            except:
                objval = None
            results = {
                "instance": instance.name,
                "nL": instance.Lcols,
                "nF": instance.Fcols,
                "mL": len(instance.a),
                "mF": len(instance.b),
                "num_threads": self.num_threads,
                "solver": "gurobi",
                "approach": None,
                "max_width": 0 if not diagram else diagram.max_width,
                "ordering_heuristic": None if not diagram else diagram.ordering_heuristic,
                "lower_bound": model.solve_details.best_bound,
                "best_obj_val": objval,
                "mip_gap": model.solve_details.gap,
                "upper_bound": objval,
                "bilevel_gap": 100 * round((objval - model.solve_details.best_bound) / abs(objval + 1e-6), 6),
                "total_runtime": None,
                "compilation_runtime": 0 if not diagram else round(diagram.compilation_runtime),
                "ordering_heuristic_runtime": 0 if not diagram else round(diagram.ordering_heuristic_runtime),
                "model_build_runtime": round(model_building_runtime),
                "model_runtime": round(model.solve_details.time),
                "num_vars": model.solve_details.columns,
                "num_constrs": "TODO",
                "nodes": None if not diagram else diagram.node_count + 2,
                "arcs": None if not diagram else diagram.arc_count,
                "follower_response": None,
                "num_cuts": "TODO",
                "cuts_runtime": "TODO",
                "SEP": "TODO",
                "solution": None
            }
            
            results["total_runtime"] = sum(value for key, value in results.items() if key in ["compilation_runtime", "model_build_runtime", "model_runtime"])

            # Retrieve vars
            try:
                sol = {
                    "x": [int(i.solution_value + 0.5) for i in vars["x"].values()],
                    "y": [int(i.solution_value + 0.5) for i in vars["y"].values()]
                }
            except:
                sol = dict()

            results["solution"] = vars

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