import logzero

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from . import constants

from .formulations.DD_formulation import get_model
from .formulations.DD_formulation_compressed_leader import get_model as get_model_with_compressed_leader

from .utils.solve_HPR import solve as solve_HPR
from .utils.solve_follower_problem import solve as solve_follower_problem
from .utils.solve_aux_problem import solve as solve_aux_problem
from .utils.build_Y import build as build_Y_set


class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def one_time_compilation_approach(self, instance, compilation_method, max_width, ordering_heuristic, solver_time_limit, build_Y=False):
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
        Y = [y]

        # Build set Y
        build_Y_runtime = 0
        if instance not in Y_tracker:
            # Collect y's
            Y, build_Y_runtime = build_Y_set(instance, num_solutions=constants.BUILD_Y_LENGTH)
            Y_tracker[instance] = {"Y": Y, "runtime": build_Y_runtime}
        else:
            Y = Y_tracker[instance]["Y"]
            build_Y_runtime = Y_tracker[instance]["runtime"]
        Y.append(y)

        # Compile diagram
        diagram = diagram_manager.compile_diagram(
            diagram, instance, compilation_method, max_width, 
            ordering_heuristic, HPR_optimal_response, Y
        )

        # Solve reformulation
        if compilation_method == "follower_then_compressed_leader":
            result, _ = self.run_DD_reformulation_with_compressed_leader(
                instance, diagram, time_limit=solver_time_limit - build_Y_runtime
            )
        else:
            result, _ = self.run_DD_reformulation(
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

        return result

    def run_DD_reformulation(self, instance, diagram, time_limit, incumbent=dict()):
        # Solve DD reformulation
        self.logger.info("Solving DD refomulation. Time limit: {} s".format(time_limit))
        model, vars = get_model(instance, diagram, time_limit, incumbent)
        model.optimize()
        self.logger.info("DD reformulation solved succesfully. Time elapsed: {} s".format(model.runtime))
        self.logger.debug("LB: {}, MIPGap: {}".format(model.objBound, model.MIPGap))
        
        # # Solve continuous relaxation
        # self.logger.info("Solving DD refomulation relaxation.".format(time_limit))
        # relaxed_model = model.relax()
        # relaxed_model.Params.OutputFlag = 0
        # relaxed_model.optimize()
        # self.logger.info("DD reformulation relaxation solved succesfully. Time elapsed: {} sec".format(relaxed_model.runtime))

        x = [i.X for i in vars["x"].values()]
        y = [i.X for i in vars["y"].values()]

        results = self.get_results(instance, diagram, model, vars)

        return results, vars

    def run_DD_reformulation_with_compressed_leader(self, instance, diagram, time_limit, incumbent=dict()):
        # Solve DD reformulation with compressed leader layers
        self.logger.info("Solving DD refomulation with compressed leader layers. Time limit: {} sec".format(time_limit))
        model, vars = get_model_with_compressed_leader(instance, diagram, time_limit, incumbent)
        model.optimize()
        self.logger.info("DD reformulation with compressed leader layers solved succesfully. Time elapsed: {} sec.".format(model.runtime))
        self.logger.debug("LB: {}, MIPGap: {}".format(model.objBound, model.MIPGap))
        
        results = self.get_results(instance, diagram, model, vars)

        vars["x"] = [i.X for i in vars["x"].values()]
        vars["y"] = [i.X for i in vars["y"].values()]
        vars["w"] = {idx: i.X for idx, i in vars["w"].items()}

        return results, vars

    def get_results(self, instance, diagram, model, vars):
        try:
            objval = model.objVal
        except:
            objval = None
        results = {
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "objval": objval,
            "lower_bound": model.objBound,
            "MIPGap": model.MIPGap,
            # "relaxation_objval": None if not relaxed_model else relaxed_model.objVal,
            "total_runtime": round(diagram.compilation_runtime + diagram.reduce_algorithm_runtime + model.runtime),
            "compilation_runtime": round(diagram.compilation_runtime),
            "reduce_algorithm_runtime": round(diagram.reduce_algorithm_runtime),
            "model_runtime": round(model.runtime),
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs
        }
        results["instance"] = instance.name
        results["width"] = diagram.width
        results["initial_width"] = diagram.initial_width
        self.logger.debug("Results: {}".format(results))
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

        return results