import logzero
from time import time
import numpy as np
from functools import partial
from textwrap import dedent

import gurobipy as gp
import cplex

from constants import RESULT_TEMPLATE

from decision_diagram_compiler.decision_diagram_compiler import DDCompiler

from models.gurobi.DD_reformulation_compact import get_model as get_gurobi_DDref_compact
from models.gurobi.DD_reformulation_extended import get_model as get_gurobi_DDref_extended
from models.gurobi.follower_problem import get_model as get_gurobi_follower_model
from models.gurobi.aux_problem import get_model as get_gurobi_aux_model
from models.gurobi.hpr import get_model as get_gurobi_hpr_model
from models.gurobi.pessimistic_follower_problem import get_model as get_gurobi_pess_follower_model
from models.gurobi.pessimistic_blocking_problem import get_model as get_gurobi_pess_blocking_model
from models.cplex.DD_reformulation_compact import get_model as get_cplex_DDref_compact
from models.cplex.follower_problem import get_model as get_cplex_follower_model
from models.cplex.aux_problem import get_model as get_cplex_aux_model

from .gurobi_callback import Callback as GurobiCallback
from .gurobi_callback import CallbackData as GurobiCallbackData
from .cplex_callback import CplexCallback

from .utils.get_max_follower_value import get_max_follower_value


class AlgorithmsManager:
    def __init__(self, instance, num_threads, mip_solver, problem_setting):
        self.logger = logzero.logger
        self.instance = instance
        self.mip_solver = mip_solver
        self.num_threads = num_threads
        if mip_solver == "gurobi":
            self.follower_model = get_gurobi_follower_model(instance)
            self.aux_model = get_gurobi_aux_model(instance)
            self.hpr_model = get_gurobi_hpr_model(instance)
            self.follower_model.Params.OutputFlag = 0
            self.aux_model.Params.OutputFlag = 0
            self.hpr_model.Params.OutputFlag = 0
            self.follower_model.Params.Threads = num_threads
            self.aux_model.Params.Threads = num_threads
            self.hpr_model.Params.Threads = num_threads
            if problem_setting == "pessimistic":
                self.pess_follower_model = get_gurobi_pess_follower_model(instance)
                self.pess_blocking_model = get_gurobi_pess_blocking_model(instance)
                self.pess_follower_model.Params.OutputFlag = 0
                self.pess_blocking_model.Params.OutputFlag = 0
                self.pess_follower_model.Params.Threads = num_threads
                self.pess_blocking_model.Params.Threads = num_threads
        elif mip_solver == "cplex":
            self.follower_model = get_cplex_follower_model(instance)
            self.aux_model = get_cplex_aux_model(instance)
            self.follower_model.context.cplex_parameters.threads = num_threads
            self.aux_model.context.cplex_parameters.threads = num_threads
            self.follower_model.parameters.mip.display.set(0)
            self.aux_model.parameters.mip.display.set(0)
            if problem_setting == "pessimistic":
                raise NotImplementedError("Pessimistic version not implemented in CPLEX yet.")

        self.hpr_model = get_gurobi_hpr_model(instance)
        self.max_follower_value = get_max_follower_value(instance)

    def solve(self, args):
        t0          = time()
        instance    = self.instance
        time_limit  = args.time_limit
        dd_args     = {
            "problem_setting": args.problem_setting,
            "max_width": float("inf") if args.dd_max_width == -1 else args.dd_max_width,
            "encoding": args.dd_encoding,
            "ordering_heuristic": args.dd_ordering_heuristic,
            "reduce_method": args.dd_reduce_method
        }
        diagram_compiler = DDCompiler(self.logger)

        # Initialize values
        iter = 1
        lb = -float("inf")
        ub = float("inf")
        gap = float("inf")
        incumbent = dict()
        result = RESULT_TEMPLATE
        model_solve_time = 0
        cuts_time = 0
        
        # Diagram compilation
        diagram = diagram_compiler.compile(instance, dd_args)

        # Compute HPR bound
        hpr_time_limit = min(180, max(time_limit - (time() - t0), 0))
        self.hpr_model.Params.timeLimit = hpr_time_limit
        self.hpr_model.Params.OutputFlag = 0
        self.hpr_model.optimize()
        x_sol = self.hpr_model._vars["x"].X
        if self.hpr_model.status == 2:
            follower_response, _, follower_opt = self.get_follower_response(args.problem_setting, x_sol, max(time_limit - (time() - t0), 0))
            lb = max(self.hpr_model.ObjVal, lb)
            result["lower_bound"] = lb
            if follower_opt and self.check_leader_feasibility(x_sol, follower_response):
                ub = min(instance.cL @ x_sol + instance.cF @ follower_response, ub)
                gap = 100 * round((ub - lb) / abs(ub + 1e-6), 6)
                incumbent = {"x": x_sol, "y": follower_response}
                result["upper_bound"] = ub
                result["bilevel_gap"] = gap
        model_solve_time += self.hpr_model.runtime
        
        if gap > 1e-6:
            # Get DD relaxation
            model = self.get_DD_reformulation(diagram, args.problem_type, args.problem_setting, incumbent)
            if self.mip_solver == "gurobi":
                model.Params.OutputFlag = 0
            elif self.mip_solver == "cplex":
                model.parameters.mip.display.set(0)

            # Main algorithm
            while time() - t0 <= time_limit:
                # Solve DD relaxation
                self.logger.debug("Solving relaxation. Time elapsed = {}s".format(round(time() - t0)))
                if self.mip_solver == "gurobi":
                    model.Params.TimeLimit = max(time_limit - (time() - t0), 0)
                    model.optimize(lambda model, where: model._cbfunc(model, where))
                    if model.status == 3:
                        self.logger.warning("Problem is infeasible!")
                        break
                    x_sol = model._vars["x"].X
                    model_solve_time += model.runtime
                    lb = max(model.ObjBound, lb)
                elif self.mip_solver == "cplex":
                    model.parameters.timelimit = max(time_limit - (time() - t0), 0)
                    model.solve(clean_before_solve=True)
                    x_sol = np.array([model._vars["x"][j].solution_value for j in range(instance.nL)])
                    model_solve_time += model.solve_details.time
                    lb = max(model.solve_details.best_bound, lb)

                # Get follower response
                self.logger.debug("Computing follower response. Time elapsed = {}s".format(round(time() - t0)))
                follower_response, follower_runtime, follower_opt = self.get_follower_response(args.problem_setting, x_sol, max(time_limit - (time() - t0), 0))
                cuts_time += follower_runtime

                # Update bounds
                if follower_opt:
                    if self.check_leader_feasibility(x_sol, follower_response):
                        ub = min(instance.cL @ x_sol + instance.cF @ follower_response, ub)
                        new_gap = 100 * round((ub - lb) / abs(ub + 1e-6), 6)
                        if new_gap < gap:
                            self.logger.warning("Gap update -> {}%. Time elapsed = {}s".format(new_gap, round(time() - t0)))
                            gap = new_gap

                # Stopping criterion
                if gap <= 1e-6:
                    self.logger.warning("Bilevel solution found!")
                    result = self.get_results(diagram, model)
                    break

                # Current solution is not bilevel feasible. Refine model
                if follower_opt:
                    cuts_time += self.add_DD_cuts(args.problem_setting, model, follower_response)

                iter += 1

        # Update final result
        result["lower_bound"] = lb
        result["upper_bound"] = ub
        result["bilevel_gap"] = round(gap, 2)
        result["model_solve_runtime"] = round(model_solve_time)
        result["cuts_runtime"] = round(cuts_time)
        result["total_runtime"] = round(time() - t0)
        result["iters"] = iter
        if result["bilevel_gap"] < 1e-6:
            result["opt"] = 1
        else:
            result["opt"] = 0
        result["time_limit"] = time_limit
        result["HPR_bound"] = self.hpr_model.ObjVal
        result["HPR_runtime"] = round(self.hpr_model.runtime)
        
        # # Compute continuous relaxation bound
        # if self.mip_solver == "gurobi":
        #     relaxed_model = model.relax()
        #     relaxed_model.Params.OutputFlag = 0
        #     relaxed_model.optimize()
        #     result["relaxation_obj_val"] = relaxed_model.objval
        # elif self.mip_solver == "cplex":
        #     model.solve(relax=True)
        #     result["relaxation_obj_val"] = model.objective_value

        self.logger.warning(dedent(
            """
            Results for {instance}: 
            Lower bound     = {lower_bound}
            Upper bound     = {upper_bound}
            Gap             = {bilevel_gap}%
            HPR bound       = {HPR_bound}
            Total runtime   = {total_runtime}
            DD width        = {dd_width}
            Iters/DD cuts   = {iters}""".format(**result)
        ).strip())

        return result
    
    def get_DD_reformulation(self, diagram, problem_type, problem_setting, incumbent=dict()):
        # Get model
        if diagram != None:
            if self.mip_solver == "gurobi":
                if diagram.encoding == "compact":
                    model = get_gurobi_DDref_compact(
                        self.instance, diagram, self.max_follower_value, 
                        problem_type, problem_setting, incumbent
                    )
                elif diagram.encoding == "extended":
                    raise NotImplementedError("Extended DD reformulation not implemented for Gurobi yet.")
                
            elif self.mip_solver == "cplex":
                if diagram.encoding == "compact":
                    model = get_cplex_DDref_compact(
                        self.instance, diagram, self.max_follower_value, 
                        problem_type, problem_setting, incumbent
                    )
                elif diagram.encoding == "extended":
                    raise NotImplementedError("Extended DD reformulation not implemented for CPLEX yet.")
        else:
            model = get_gurobi_hpr_model(self.instance)
                
        # Add DD cut associated to incumbent
        if incumbent:
            model._build_time += self.add_DD_cuts(problem_setting, model, incumbent["y"])

        if self.mip_solver == "gurobi":
            callback_data = GurobiCallbackData(self.instance)
            callback_func = partial(GurobiCallback().callback, cbdata=callback_data)
            model._cbdata = callback_data
            model._cbfunc = callback_func
            model.Params.Threads = self.num_threads
            
        elif self.mip_solver == "cplex":
            callback = CplexCallback(self.instance, model)
            model._cbfunc = callback
            model.parameters.threads = self.num_threads
            
        return model

    def get_follower_response(self, problem_setting, x_sol, time_limit):
        t0 = time()
        follower_value = None
        follower_response = None
        opt = False

        follower_model = self.follower_model
        aux_model = self.aux_model
        if problem_setting == "pessimistic":
            pess_blocking_model = self.pess_blocking_model
            pess_follower_model = self.pess_follower_model

        # Solve models
        if self.mip_solver == "gurobi":
            # Solve follower model
            follower_model._constrs.RHS = self.instance.b - self.instance.C @ x_sol
            follower_model.Params.TimeLimit = max(0, time_limit - (time() - t0))
            follower_model.optimize()
            if follower_model.status == 2:
                follower_value = follower_model.ObjVal
                if problem_setting == "optimistic":
                    opt = True
                    follower_response = follower_model._vars["y"].X
                    # Solve aux model
                    self.aux_model._constrs["leader_constrs"].RHS = self.instance.a - self.instance.A @ x_sol
                    self.aux_model._constrs["follower_constrs"].RHS = self.instance.b - self.instance.C @ x_sol
                    self.aux_model._constrs["vf_bound"].RHS = follower_value + 1e-8
                    aux_model.Params.TimeLimit = max(0, time_limit)
                    aux_model.optimize()
                    if aux_model.status == 2:
                        follower_response = aux_model._vars["y"].X
                elif problem_setting == "pessimistic":
                    # Check if leader solution can be blocked by follower
                    pess_blocking_model._constrs["follower_constrs"].RHS = self.instance.b - self.instance.C @ x_sol
                    pess_blocking_model._constrs["vf_bound"].RHS = follower_value + 1e-8
                    for i in range(self.instance.mL):
                        pess_blocking_model._constrs["leader_blocking"][i].RHS = self.instance.a[i] - self.instance.A[i] @ x_sol + 1
                    pess_blocking_model.Params.TimeLimit = max(0, time_limit - (time() - t0))
                    pess_blocking_model.optimize()
                    # Follower cannot block
                    if pess_blocking_model.status not in [2, 9]:
                        pess_follower_model._constrs["follower_constrs"].RHS = self.instance.b - self.instance.C @ x_sol
                        pess_follower_model._constrs["vf_bound"].RHS = follower_value + 1e-8
                        pess_follower_model.Params.TimeLimit = max(0, time_limit - (time() - t0))
                        pess_follower_model.optimize()
                        if pess_follower_model.status == 2:
                            opt = True
                            follower_response = pess_follower_model._vars["y"].X
                    # Follower can block
                    else:
                        opt = True
                        follower_response = pess_blocking_model._vars["y"].X
            
        elif self.mip_solver == "cplex":
            if problem_setting == "pessimistic":
                raise NotImplementedError("Pessimistic setting not implemented in CPLEX yet!")
            
            # Solve follower model
            for i in range(self.instance.mF):
                self.follower_model._constrs[i].rhs = float(self.instance.b[i] - self.instance.C[i] @ x_sol)
            follower_model.parameters.timelimit = max(0, time_limit - (time() - t0))
            follower_model.solve(clean_before_solve=True)
            try:
                opt = True
                follower_value = follower_model.objective_value
                follower_response = np.array([round(follower_model._vars["y"][j].solution_value) for j in range(self.instance.nF)])
                # Solve aux model
                for i in range(self.instance.Frows):
                    self.aux_model._constrs["HPR"][i].rhs = float(self.instance.b[i] - self.instance.C[i] @ x_sol)
                self.aux_model._constrs["objval"].rhs = follower_value + 1e-6
                aux_model.parameters.timelimit = max(0, time_limit - (time() - t0))
                aux_model.solve(clean_before_solve=True)
                try:
                    follower_response = np.array([round(aux_model._vars["y"][j].solution_value) for j in range(self.instance.nF)])
                except:
                    pass
            except:
                pass

        runtime = time() - t0

        return follower_response, runtime, opt
    
    def add_DD_cuts(self, problem_setting, model, y_sol):
        t0 = time()
        instance = self.instance
        interaction_rows = [i for i in range(instance.mF) if instance.interaction[i] == "both"]

        if self.mip_solver == "gurobi":
            # Create vars
            alpha = model.addVar(vtype=gp.GRB.BINARY)
            beta = model.addVars(interaction_rows, vtype=gp.GRB.BINARY)
            
            # Alpha-beta relationship
            model.addConstr(alpha >= gp.quicksum(1 - beta[i] for i in interaction_rows) - (len(interaction_rows) - 1))

            # Blocking definition
            M = {i: sum(min(instance.C[i][j], 0) for j in range(instance.nL)) for i in interaction_rows}
            model.addConstrs(
                instance.C[i] @ model._vars["x"] >= M[i] + beta[i] * (-M[i] + instance.b[i] - instance.D[i] @ y_sol + 0.5) 
                for i in interaction_rows
            )  

            # Value-function bound
            M = self.max_follower_value - instance.d @ y_sol
            model.addConstr(instance.d @ model._vars["y"] <= instance.d @ y_sol + M * (1 - alpha))

            if problem_setting == "pessimistic":
                v = model.addVar(vtype=gp.GRB.BINARY)
                model.addConstr(instance.d @ model._vars["y"] - 1e8 * v <= instance.d @ y_sol - 1)
                model.addConstr(instance.cF @ model._vars["y"] >= instance.cF @ y_sol - 1e8 * (1 - alpha) - 1e8 * (1 - v))
                model.addConstrs(
                    instance.A[i] @ model._vars["x"] + instance.B[i] @ y_sol <= instance.a[i] + 1e8 * (1 - alpha) + 1e8 * (1 - v)
                    for i in interaction_rows
                ) 

            # Reset model
            model.reset()

        elif self.mip_solver == "cplex":
            alpha = model.binary_var()
            beta = model.binary_var_dict(interaction_rows)

            # Alpha-beta relationship
            model.add_constraint_(alpha >= model.sum(1 - beta[i] for i in interaction_rows) - (len(interaction_rows) - 1))

            # Blocking definition
            M = {i: sum(min(instance.C[i][j], 0) for j in range(instance.nL)) for i in interaction_rows}
            model.add_constraints_(
                model.sum(instance.C[i][j] * model._vars["x"][j] for j in range(instance.nL)) 
                >= M[i] + beta[i] * (-M[i] + instance.b[i] - instance.D[i] @ y_sol + 1) 
                for i in interaction_rows
            )  

            # Value-function bound
            M = self.max_follower_value - instance.d @ y_sol
            model.add_constraint_(
                model.sum(instance.d[j] * model._vars["y"][j] for j in range(instance.nF)) 
                <= instance.d @ y_sol + M * (1 - alpha)
            )

            if problem_setting == "pessimistic":
                raise NotImplementedError("Pessimistic cuts not implemented for CPLEX yet.")

        return time() - t0

    def check_leader_feasibility(self, x, y):
        if self.instance.mL == 0:
            return True
        
        elif np.all([self.instance.A @ x + self.instance.B @ y <= self.instance.a]):
            return True
        
        return False

    def get_results(self, diagram, model):
        instance = self.instance
        result = RESULT_TEMPLATE

        # Retrieve results
        if self.mip_solver == "gurobi":
            try:
                upper_bound = model.objVal
                gap = (upper_bound - model.ObjBound) / abs(upper_bound + 1e-8)
            except:
                upper_bound = float("inf")
                gap = float("inf")
            
            # Instance
            result["instance"]                          = instance.name
            result["nL"]                                = instance.nL
            result["nF"]                                = instance.nF
            result["mL"]                                = instance.mL
            result["mF"]                                = instance.mF

            # Solver params
            result["num_threads"]                       = self.num_threads

            # Solver stats
            result["num_vars"]                          = model.numVars
            result["num_constrs"]                       = model.numConstrs
            result["BB_node_count"]                     = model.nodeCount
            result["lower_bound"]                       = model.ObjBound
            result["upper_bound"]                       = upper_bound
            result["gap"]                               = gap
            result["opt"]                               = 1 if gap <= 1e-6 else 0
            if diagram:
                result["dd_encoding"]                   = diagram.encoding
                result["dd_max_width"]                  = diagram.max_width
                result["dd_ordering_heuristic"]         = diagram.ordering_heuristic
                result["dd_reduce_method"]              = diagram.reduce_method
                result["dd_width"]                      = diagram.width
                result["dd_nodes"]                      = diagram.node_count
                result["dd_arcs"]                       = diagram.arc_count
                result["dd_ordering_heuristic_time"]    = diagram.ordering_heuristic_runtime
                result["dd_compilation_runtime"]        = diagram.compilation_runtime
            result["model_build_runtime"]               = model._build_time
            result["model_solve_runtime"]               = model.runtime
            result["total_runtime"]                     = result["dd_compilation_runtime"]\
                                                        + result["model_build_runtime"]\
                                                        + result["model_solve_runtime"]
            try:
                sol = {
                    "x": model._vars["x"].X.round(),
                    "y": model._vars["y"].X.round()
                }
            except:
                sol = dict()
            result["solution"] = sol

        ## CPLEX results
        elif self.mip_solver == "cplex":
            pass # TODO

        return result