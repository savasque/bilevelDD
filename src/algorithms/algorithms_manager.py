import logzero
from time import time
import numpy as np
from functools import partial
import gurobipy as gp

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

import constants

from .formulations.DD_formulation import get_model
from .formulations.DD_formulation_compressed_leader import get_model as get_model_with_compressed_leader

from .utils.solve_HPR import solve as solve_HPR
from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem
from .utils.sampler import Sampler


class CallbackData:
    def __init__(self, instance):
        self.instance = instance
        self.root_node_bound = None
        self.was_root_node_visited = False
        self.lazy_cuts = True

class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
        self.callback_data = None
        self.callback_func = None
    
    def one_time_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, discard_method, solver_time_limit):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()
        t0 = time()

        # Get HPR bounds
        HPR_value, HPR_optimal_solution, UB, _ = self.get_HPR_bounds(instance)
        self.logger.info("Updated bounds -> LB: {} - UB: {}".format(HPR_value, UB))

        # Build set Y
        if constants.SAMPLING_LENGTH:
            Y, sampling_runtime = self.sample_follower_solutions(instance)
            Y = [HPR_optimal_solution["y"]] + Y
        else:
            Y = [HPR_optimal_solution["y"]]
            sampling_runtime = 0

        # HPR_value = None
        # Y = []
        # sampling_runtime = 0
        # UB = float("inf")

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, discard_method, Y
        )

        # Solve reformulation
        if compilation_method == "follower_then_compressed_leader":
            result, model, _ = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit - sampling_runtime
            )
        else:
            result, model, _ = self.run_DD_reformulation(
                instance, diagram, time_limit=solver_time_limit - sampling_runtime
            )

        total_runtime = time() - t0

        # Update final result
        result["approach"] = "one_time_compilation"
        result["discard_method"] = discard_method
        result["HPR"] = HPR_value
        result["sampling"] = True if len(Y) >= 2 else False
        result["Y_length"] = len(Y)
        result["sampling_runtime"] = sampling_runtime
        result["total_runtime"] = round(total_runtime)
        result["time_limit"] = solver_time_limit
        result["num_nodes"] = diagram.node_count + 2
        result["num_arcs"] = diagram.arc_count
        result["upper_bound"] = min(result["upper_bound"], UB)
        result["bilevel_gap"] = round((result["upper_bound"] - result["lower_bound"]) / abs(result["upper_bound"] + 1e-2), 3) if result["upper_bound"] < float("inf") else None
        result["iters"] = 0

        # Compute continuous relaxation bound
        relaxed_model = model.relax()
        relaxed_model.Params.OutputFlag = 0
        relaxed_model.optimize()
        result["root_node_bound"] = self.callback_data.root_node_bound
        result["relaxation_obj_val"] = relaxed_model.objval

        self.logger.info(
            "Results for {instance} -> LB: {lower_bound} - UB: {upper_bound} - MIPGap: {mip_gap} - BilevelGap: {bilevel_gap} - HPR: {HPR} - Runtime: {total_runtime} - DDWidth: {width}".format(**result)
        )
        self.logger.debug(
            "Runtimes -> Compilation: {compilation_runtime} - Model build: {model_build_runtime} - Model solve: {model_runtime}".format(**result)
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
        Y = [HPR_optimal_solution["y"]]

        # HPR_value = None
        # Y = []
        # UB = float("inf")

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, discard_method, Y
        )

        # Solve reformulation
        if compilation_method == "follower_then_compressed_leader":
            result, model, vars = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit
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
        result["sampling_runtime"] = 0
        result["total_runtime"] = round(total_runtime + instance.load_runtime)
        result["time_limit"] = solver_time_limit
        result["num_nodes"] = diagram.node_count + 2
        result["num_arcs"] = diagram.arc_count
        result["upper_bound"] = min(result["upper_bound"], UB)
        result["bilevel_gap"] = round((result["upper_bound"] - result["lower_bound"]) / abs(result["upper_bound"] + 1e-2), 3) if result["upper_bound"] < float("inf") else None
        result["iters"] = iter

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
    
    def sample_follower_solutions(self, instance):
        sampler = Sampler(self.logger, sampling_method=constants.SAMPLING_METHOD)
        # Collect y's
        Y, sampling_runtime = sampler.sample(instance, constants.SAMPLING_LENGTH)

        return Y, sampling_runtime
    
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

        # # Capacity constrs
        # model.addConstrs(w[arc.id, diagram.id] <= vars["y"][j] for j in vars["y"] for arc in diagram.arcs if arc.player == "follower" and arc.var_index == j and arc.value == 1)
        # model.addConstrs(w[arc.id, diagram.id] <= 1 - vars["y"][j] for j in vars["y"] for arc in diagram.arcs if arc.player == "follower" and arc.var_index == j and arc.value == 0)

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
        model.addConstrs(instance.C[i] @ vars["x"].values() >= -M_blocking[i] + beta[arc.id, diagram.id, i] * (M_blocking[i] + arc.block_values[i]) for arc in diagram.arcs if arc.player == "leader" for i in interaction_indices)     

    def run_DD_reformulation(self, instance, diagram, time_limit, incumbent=dict()):
        # Solve DD reformulation
        self.logger.info("Solving DD refomulation. Time limit: {} s".format(time_limit))
        model, vars = get_model(instance, diagram, time_limit, incumbent)
        model.optimize()
        self.logger.info("DD reformulation solved succesfully. Time elapsed: {} s".format(model.runtime))
        self.logger.debug("LB: {}, MIPGap: {}".format(model.objBound, model.MIPGap))

        results = self.get_results(instance, diagram, model, vars)

        return results, model, vars

    def run_DD_reformulation_with_compressed_leader(self, instance, diagram, time_limit, incumbent=dict()):
        # Callback function
        self.callback_data = CallbackData(instance)
        callback_func = lambda model, where: self.callback(model, where, self.callback_data)

        # Solve DD reformulation
        self.logger.info("Solving DD refomulation -> Time limit: {} s".format(time_limit))
        model, vars, model_building_runtime = get_model_with_compressed_leader(instance, diagram, time_limit, incumbent)
        model.Params.LazyConstraints = 1
        model.optimize(callback_func)
        self.logger.info("DD reformulation succesfully solved -> Time elapsed: {} s".format(model.runtime))

        results = self.get_results(instance, diagram, model, vars, model_building_runtime)

        feas, msg = self.check_solution_feasibility(instance, results["vars"])
        if not feas:
            raise ValueError("Solution is infeasible -> {}".format(msg))

        return results, model, vars
    
    def callback(self, model, where, cbdata):
        if where == gp.GRB.Callback.MIPNODE:
            if not cbdata.was_root_node_visited:
                cbdata.root_node_bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)
                cbdata.was_root_node_visited = True
        
        # elif where == gp.GRB.Callback.MIPSOL and cbdata.lazy_cuts:
        #     # Retrieve solution
        #     x = model._vars["x"]
        #     y = model._vars["y"]
        #     x_sol = list(model.cbGetSolution(x).values())
        #     follower_value = solve_follower_problem(cbdata.instance, x_sol)[0]
        #     follower_response = solve_aux_problem(cbdata.instance, x_sol, follower_value)[1]

        #     # Add cut    

    def get_results(self, instance, diagram, model, vars, model_building_runtime):
        try:
            objval = model.objVal
        except:
            objval = None
        results = {
            "approach": None,
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "lower_bound": model.ObjBound,
            "best_obj_val": objval,
            "mip_gap": model.MIPGap,
            "upper_bound": float("inf"),
            "bilevel_gap": None,
            "total_runtime": None,
            "data_load_runtime": round(instance.load_runtime),
            "compilation_runtime": round(diagram.compilation_runtime),
            "reduce_algorithm_runtime": round(diagram.reduce_algorithm_runtime),
            "model_build_runtime": round(model_building_runtime),
            "model_runtime": round(model.runtime),
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs,
            "num_nodes": diagram.node_count,
            "num_arcs": diagram.arc_count,
            "num_node_merges": diagram.num_merges
        }
        results["instance"] = instance.name
        results["width"] = diagram.width
        results["initial_width"] = diagram.initial_width
        try:
            vars = {
                "x": [int(i.X + 0.5) for i in vars["x"].values()],
                "y": [int(i.X + 0.5) for i in vars["y"].values()],
                "w": [key for key, value in vars["w"].items() if value.X > 0],
            }
        except:
            vars = dict()

        # Compute upper bound
        follower_value = solve_follower_problem(instance, vars["x"])[0]
        follower_response = solve_aux_problem(instance, vars["x"], follower_value)[1]
        if instance.d @ vars["y"] <= instance.d @ follower_response:
            # Current solution is bilevel optimal
            results["upper_bound"] = instance.c_leader @ vars["x"] + instance.c_follower @ vars["y"]
        else:
             # Current solution allows to compute an upper bound
            if instance.Lrows:
                if np.all([instance.A @ vars["x"] + instance.B @ follower_response <= instance.a]):
                    results["upper_bound"] = instance.c_leader @ vars["x"] + instance.c_follower @ follower_response
            else:
                results["upper_bound"] = instance.c_leader @ vars["x"] + instance.c_follower @ follower_response
        
        results["vars"] = vars
        results["opt_y"] = follower_response

        return results
    
    def check_solution_feasibility(self, instance, solution):
        tol = 1e-6
        if instance.A and not np.all(instance.A @ solution["x"] + instance.B @ solution["y"] <= instance.a + tol):
            return False, "leader_infeasible"
        if not np.all(instance.C @ solution["x"] + instance.D @ solution["y"] <= instance.b + tol):
            return False, "follower_infeasible"
        
        return True, "feasible"