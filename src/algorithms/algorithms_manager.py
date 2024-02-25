import logzero
from time import time
import numpy as np

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

import constants

from .formulations.DD_formulation import get_model
from .formulations.DD_formulation_compressed_leader import get_model as get_model_with_compressed_leader

from .utils.solve_HPR import solve as solve_HPR
from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem
from .utils.sampler import Sampler


class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def one_time_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, discard_method, solver_time_limit):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()

        UB = float("inf")

        # Get HPR bound
        self.logger.info("Solving HPR")
        HPR_value, HPR_solution = solve_HPR(instance)
        _, y = solve_follower_problem(instance, HPR_solution["x"])  # Optimal follower response
        _, y = solve_aux_problem(instance, HPR_solution["x"], instance.d @ y)  # Break ties among optimal follower responses
        self.logger.info("HPR solved. ObjVal: {}".format(HPR_value))
        HPR_optimal_response = {"x": HPR_solution["x"], "y": y}

        # Get upper bound
        if instance.Lrows:
            if np.all([instance.A @ HPR_optimal_response["x"] + instance.B @ HPR_optimal_response["y"] <= instance.a]):
                UB = instance.c_leader @ HPR_optimal_response["x"] + instance.c_follower @ HPR_optimal_response["y"]
                self.logger.info("Initial UB given by HPR: {}".format(UB))
        else:
            UB = instance.c_leader @ HPR_optimal_response["x"] + instance.c_follower @ HPR_optimal_response["y"]
            self.logger.info("Initial UB given by HPR: {}".format(UB))

        # Build set Y
        sampling_runtime = 0
        sampler = Sampler(self.logger, sampling_method=constants.SAMPLING_METHOD)
        if constants.BUILD_Y_LENGTH:
            # Collect y's
            Y, sampling_runtime = sampler.sample(instance, constants.BUILD_Y_LENGTH)
            Y.append(y)
        else:
            Y = [y]

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, discard_method, Y
        )

        # Solve reformulation
        if compilation_method == "follower_then_compressed_leader":
            result, _, _ = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit - sampling_runtime
            )
        else:
            result, _, _ = self.run_DD_reformulation(
                instance, diagram, time_limit=solver_time_limit - sampling_runtime
            )

        # Update final result
        result["approach"] = "one_time_compilation"
        result["discard_method"] = discard_method
        result["HPR"] = HPR_value
        result["Y_length"] = len(Y)
        result["sampling_runtime"] = sampling_runtime
        result["total_runtime"] += sampling_runtime
        result["time_limit"] = solver_time_limit
        result["num_nodes"] = diagram.node_count + 2
        result["num_arcs"] = diagram.arc_count
        result["upper_bound"] = min(result["upper_bound"], UB)
        result["bilevel_gap"] = round((result["upper_bound"] - result["obj_val"]) / abs(result["upper_bound"] + 1e-2), 3) if result["upper_bound"] < float("inf") else None
        result["iters"] = 0

        self.logger.info("Results for {instance}: ObjVal: {obj_val} - UB: {upper_bound} - MIPGap: {mip_gap} - BilevelGap: {bilevel_gap} - HPR: {HPR} - Runtime: {total_runtime} - DDWidth: {width}".format(**result))

        del result["vars"]
        del result["opt_y"]

        return result

    def iterative_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, discard_method, solver_time_limit):
        diagram = DecisionDiagram(0)
        diagram_manager = DecisionDiagramManager()

        t0 = time()
        iter = 0
        UB = float("inf")
        LBs = list()

        # Get HPR bound
        self.logger.info("Solving HPR")
        HPR_value, HPR_solution = solve_HPR(instance)
        _, y = solve_follower_problem(instance, HPR_solution["x"])  # Optimal follower response
        _, y = solve_aux_problem(instance, HPR_solution["x"], instance.d @ y)  # Break ties among optimal follower responses
        self.logger.info("HPR solved. ObjVal: {}".format(HPR_value))
        HPR_optimal_response = {"x": HPR_solution["x"], "y": y}
        Y = [y]
        
        # Get upper bound
        if instance.Lrows:
            if np.all([instance.A @ HPR_optimal_response["x"] + instance.B @ HPR_optimal_response["y"] <= instance.a]):
                UB = instance.c_leader @ HPR_optimal_response["x"] + instance.c_follower @ HPR_optimal_response["y"]
                self.logger.info("Initial UB given by HPR: {}".format(UB))
        else:
            UB = instance.c_leader @ HPR_optimal_response["x"] + instance.c_follower @ HPR_optimal_response["y"]
            self.logger.info("Initial UB given by HPR: {}".format(UB))

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, discard_method, Y
        )

        if compilation_method == "follower_then_compressed_leader":
            result, model, vars = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit
            )
        else:
            result, model, vars = self.run_DD_reformulation(
                instance, diagram, time_limit=solver_time_limit
            )

        model.Params.OutputFlag = 0
        LBs.append(result["obj_val"])

        # Create new DDs and cuts
        while time() - t0 <= solver_time_limit:
            if instance.d @ result["vars"]["y"] <= instance.d @ result["opt_y"]:
                UB = instance.c_leader @ result["vars"]["x"] + instance.c_follower @ result["vars"]["y"]
                self.logger.info("Bilevel solution found - Objval: {} - UB: {}".format(result["obj_val"], UB))
                break
            else:
                # Current solution is not b-feasible. Add a cut and solve again
                # Build extra DD
                diagram = DecisionDiagram(iter + 1)
                Y = [result["opt_y"]]
                diagram = diagram_manager.compile_diagram(
                    diagram, instance, compilation_method, float("inf"), 
                    ordering_heuristic, discard_method, Y,
                    skip_brute_force_compilation=True
                )
                # Add associated cuts and solve
                self.add_DD_cuts(instance, diagram, model, vars)
                model.Params.TimeLimit = solver_time_limit - (time() - t0)
                self.logger.info("Solving new model with added cuts. Time limit: {} s".format(solver_time_limit - (time() - t0)))
                model.optimize()
                if model.ObjVal >= result["obj_val"]:
                    self.logger.info("Model succesfully solved. Time elapsed: {} s".format(model.runtime))
                    self.logger.info("New LB: {}".format(model.ObjVal))
                    result = self.get_results(instance, diagram, model, vars)

            # Tracking
            UB = min(UB, result["upper_bound"])
            LBs.append(result["obj_val"])
            iter += 1

        # Update results
        result["approach"] = "iterative_cuts"
        result["discard_method"] = discard_method
        result["HPR"] = HPR_value
        result["Y_length"] = 0
        result["sampling_runtime"] = 0
        result["max_width"] = max_width
        result["time_limit"] = solver_time_limit
        result["num_nodes"] = diagram.node_count + 2
        result["num_arcs"] = diagram.arc_count
        result["upper_bound"] = min(result["upper_bound"], UB)
        result["bilevel_gap"] = round((result["upper_bound"] - result["obj_val"]) / abs(result["upper_bound"] + 1e-2), 3) if result["upper_bound"] < float("inf") else None
        result["iters"] = iter
        result["total_runtime"] = time() - t0
        del result["vars"]
        del result["opt_y"]

        return result
    
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
        # Solve DD reformulation with compressed leader layers
        self.logger.info("Solving DD refomulation with compressed leader layers. Time limit: {} sec".format(time_limit))
        model, vars = get_model_with_compressed_leader(instance, diagram, time_limit, incumbent)
        model.optimize()
        self.logger.info("DD reformulation with compressed leader layers solved succesfully. Time elapsed: {} sec.".format(model.runtime))
        
        results = self.get_results(instance, diagram, model, vars)

        return results, model, vars

    def get_results(self, instance, diagram, model, vars):
        try:
            objval = model.objVal
        except:
            objval = None
        results = {
            "approach": None,
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "obj_val": objval,
            "mip_gap": model.MIPGap,
            "upper_bound": float("inf"),
            "bilevel_gap": None,
            "total_runtime": round(diagram.compilation_runtime + diagram.reduce_algorithm_runtime + model.runtime),
            "compilation_runtime": round(diagram.compilation_runtime),
            "reduce_algorithm_runtime": round(diagram.reduce_algorithm_runtime),
            "model_runtime": round(model.runtime),
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs,
            "nodes": diagram.node_count,
            "arcs": diagram.arc_count
        }
        results["instance"] = instance.name
        results["width"] = diagram.width
        results["initial_width"] = diagram.initial_width
        try:
            vars = {
                "x": [i.X for i in vars["x"].values()],
                "y": [i.X for i in vars["y"].values()],
                "w": {key: value.X for key, value in vars["w"].items()}
            }
        except:
            vars = dict()

        follower_value = solve_follower_problem(instance, vars["x"])[0]
        follower_response = solve_aux_problem(instance, vars["x"], follower_value)[1]

        # Compute upper bound
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