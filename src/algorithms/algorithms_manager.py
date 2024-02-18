import logzero
from time import time

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

import constants

from .formulations.DD_formulation import get_model
from .formulations.DD_formulation_compressed_leader import get_model as get_model_with_compressed_leader

from .utils.solve_HPR import solve as solve_HPR
from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem
from .utils.build_Y import build as build_Y_set


class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def one_time_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, solver_time_limit):
        diagram = DecisionDiagram()
        diagram_manager = DecisionDiagramManager()

        Y_tracker = dict()

        # Get HPR bound
        self.logger.info("Solving HPR")
        HPR_value, HPR_solution = solve_HPR(instance)
        _, y = solve_follower_problem(instance, HPR_solution["x"])  # Optimal follower response
        _, y = solve_aux_problem(instance, HPR_solution["x"], instance.d @ y)  # Break ties among optimal follower responses
        self.logger.info("HPR solved. ObjVal: {}".format(HPR_value))
        HPR_optimal_response = {"x": HPR_solution["x"], "y": y}

        # Build set Y
        build_Y_runtime = 0
        if constants.BUILD_Y_LENGTH:
            if instance not in Y_tracker:
                # Collect y's
                Y, build_Y_runtime = build_Y_set(instance, num_solutions=constants.BUILD_Y_LENGTH)
                Y_tracker[instance] = {"Y": Y, "runtime": build_Y_runtime}
            else:
                Y = Y_tracker[instance]["Y"]
                build_Y_runtime = Y_tracker[instance]["runtime"]
            Y.append(y)
        else:
            Y = [y]

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, HPR_optimal_response, Y
        )

        # Solve reformulation
        if compilation_method == "follower_then_compressed_leader":
            result, _, _ = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit - build_Y_runtime
            )
        else:
            result, _, _ = self.run_DD_reformulation(
                instance, diagram, time_limit=solver_time_limit - build_Y_runtime
            )

        # Update final result
        result["HPR"] = HPR_value
        result["Y_length"] = len(Y)
        result["build_Y_runtime"] = build_Y_runtime
        result["total_runtime"] += build_Y_runtime
        result["time_limit"] = solver_time_limit
        result["num_nodes"] = diagram.node_count + 2
        result["num_arcs"] = diagram.arc_count

        self.logger.debug("Results: ObjVal: {obj_val} - UB: {upper_bound} - MIPGap: {mip_gap} - BilevelGap: {bilevel_gap} - Runtime: {total_runtime} - DDWidth: {width}".format(**result))

        return result

    def iterative_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, solver_time_limit):
        diagram = DecisionDiagram()
        diagram_manager = DecisionDiagramManager()

        t0 = time()
        UB = float("inf")

        # Get HPR bound
        self.logger.info("Solving HPR")
        HPR_value, HPR_solution = solve_HPR(instance)
        _, y = solve_follower_problem(instance, HPR_solution["x"])  # Optimal follower response
        _, y = solve_aux_problem(instance, HPR_solution["x"], instance.d @ y)  # Break ties among optimal follower responses
        self.logger.info("HPR solved. ObjVal: {}".format(HPR_value))
        HPR_optimal_response = {"x": HPR_solution["x"], "y": y}
        Y = [y]

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, HPR_optimal_response, Y
        )

        if compilation_method == "follower_then_compressed_leader":
            result, model, vars = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit
            )
        else:
            result, model, vars = self.run_DD_reformulation(
                instance, diagram, time_limit=solver_time_limit
            )
        
        t1 = time()

        model.Params.OutputFlag = 0
        LBs = [result["ObjVal"]]
        Ys = [result["opt_y"]]
        UBs = [result["upper_bound"]]

        # Create new DDs and cuts
        while time() - t0 <= solver_time_limit - (t1 - t0):
            if instance.d @ result["vars"]["y"] <= instance.d @ result["opt_y"]:
                self.logger.info("Bilevel solution found - Objval: {}".format(result["ObjVal"]))
                break
            else:
                # Current solution is not b-feasible. Add a cut and solve again
                # Build extra DD
                diagram = DecisionDiagram()
                Y = [result["opt_y"]]
                diagram = diagram_manager.compile_diagram(
                    diagram, instance, compilation_method, 0, 
                    ordering_heuristic, HPR_optimal_response, Y
                )
                # Add associated cuts and solve
                self.add_DD_cuts(instance, diagram, model, vars)
                model.optimize()
                self.logger.info("New LB: {}".format(model.ObjVal))
                result = self.get_results(instance, diagram, model, vars)

            # Tracking
            LBs.append(result["ObjVal"])
            Ys.append(result["opt_y"])
            UBs.append(result["upper_bound"])

        result["LBs"] = LBs
        result["upper_bound"] = min(UBs)

        # Results fix
        result["max_width"] = max_width

        return result
    
    def add_DD_cuts(self, instance, diagram, model, vars):
        import gurobipy as gp
        import numpy as np

        interaction_indices = [i for i in range(instance.Frows) if np.any(instance.C[i]) and np.any(instance.D[i])]

        # Reset model
        model.reset()

        # Create vars
        w = model.addVars([arc.id for arc in diagram.arcs], ub=1, name="w")
        pi = model.addVars([node.id for node in diagram.nodes.values()], lb=-gp.GRB.INFINITY, name="pi")
        gamma = model.addVars([arc.id for arc in diagram.arcs if arc.player == "leader"], name="gamma")
        alpha = model.addVars([arc.id for arc in diagram.arcs if arc.player == "leader"], vtype=gp.GRB.BINARY, name="alpha")
        beta = model.addVars([arc.id for arc in diagram.arcs if arc.player == "leader"], interaction_indices, vtype=gp.GRB.BINARY, name="beta")
        
        # Value-function constr
        model.addConstr(gp.quicksum(instance.d[j] * vars["y"][j] for j in range(instance.Fcols)) <= gp.quicksum(arc.cost * w[arc.id] for arc in diagram.arcs), name="ValueFunction")

        # Flow constrs
        model.addConstr(gp.quicksum(w[arc.id] for arc in diagram.nodes["root"].outgoing_arcs) - gp.quicksum(w[arc.id] for arc in diagram.nodes["root"].incoming_arcs) == 1, name="FlowRoot")
        model.addConstr(gp.quicksum(w[arc.id] for arc in diagram.nodes["sink"].outgoing_arcs) - gp.quicksum(w[arc.id] for arc in diagram.nodes["sink"].incoming_arcs) == -1, name="FlowSink")
        model.addConstrs(gp.quicksum(w[arc.id] for arc in node.outgoing_arcs) - gp.quicksum(w[arc.id] for arc in node.incoming_arcs) == 0 for node in diagram.nodes.values() if node.id not in ["root", "sink"])

        # Capacity constrs
        model.addConstrs(w[arc.id] <= vars["y"][j] for j in vars["y"] for arc in diagram.arcs if arc.player == "follower" and arc.var_index == j and arc.value == 1)
        model.addConstrs(w[arc.id] <= 1 - vars["y"][j] for j in vars["y"] for arc in diagram.arcs if arc.player == "follower" and arc.var_index == j and arc.value == 0)

        # Dual feasibility
        model.addConstrs((pi[arc.tail] - pi[arc.head] <= arc.cost for arc in diagram.arcs if arc.player in ["follower", None]), name="DualFeasFollower")
        model.addConstrs((pi[arc.tail] - pi[arc.head] - gamma[arc.id] <= 0 for arc in diagram.arcs if arc.player == "leader"), name="DualFeasLeader")

        # Strong duality
        model.addConstr(pi[diagram.nodes["root"].id] == gp.quicksum(arc.cost * w[arc.id] for arc in diagram.arcs), name="StrongDualRoot")
        model.addConstr(pi[diagram.nodes["sink"].id] == 0, name="StrongDualSink")

        # Gamma bounds
        M_gamma = 1e6
        model.addConstrs(gamma[arc.id] <= M_gamma * alpha[arc.id] for arc in diagram.arcs if arc.player == "leader")

        # Alpha-beta relationship
        model.addConstrs(alpha[arc.id] <= gp.quicksum(beta[arc.id, i] for i in interaction_indices) for arc in diagram.arcs if arc.player == "leader")

        # Blocking definition
        M_blocking = {i: sum(max(-instance.C[i][j], 0) for j in range(instance.Lcols)) for i in range(instance.Frows)}
        model.addConstrs(instance.C[i] @ vars["x"].values() >= -M_blocking[i] + beta[arc.id, i] * (M_blocking[i] + arc.block_values[i]) for arc in diagram.arcs if arc.player == "leader" for i in interaction_indices)

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
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "obj_val": objval,
            "mip_gap": model.MIPGap,
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
        results["upper_bound"] = instance.c_leader @ vars["x"] + instance.c_follower @ follower_response
        results["bilevel_gap"] = round((results["upper_bound"] - results["obj_val"]) / abs(results["upper_bound"]), 3)

        return results