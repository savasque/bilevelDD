import logging, logzero
import argparse

import constants
from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager
from algorithms.utils.solve_HPR import solve as solve_HPR
from algorithms.utils.solve_follower_problem import solve as solve_follower_problem
from algorithms.utils.solve_aux_problem import solve as solve_aux_problem

# LogLevel
logzero.loglevel(logging.getLevelName(constants.LOG_LEVEL))
logger = logzero.logger

def run(args):
    # Classes instantiation
    parser = Parser()
    algorithms_manager = AlgorithmsManager()

    Y_tracker = dict()
    collect_Y_runtime = 0

    # Simulation
    if type(args.instance_name) != list:
        instances = [args.instance_name]
    else:
        instances = args.instance_name
    for instance_name in instances:
        for max_width in constants.MAX_WIDTH:
            for ordering_heuristic in constants.ORDERING_HEURISTIC:
                for compilation_method in constants.COMPILATION_METHOD: 
                    # Load data
                    instance = parser.build_instance(instance_name)

                    # # Iterative compilation approach
                    # if compilation_method == "iterative":
                    #     time_limit = int(constants.SOLVER_TIME_LIMIT)
                    #     best_result = {"lower_bound": -float("inf")}
                    #     lb_tracking = list()
                    #     total_runtime = 0
                    #     compilation_runtime = 0
                    #     reduce_algorithm_runtime = 0
                    #     model_runtime = 0

                    #     # Collect solution
                    #     x = vars["x"]
                    #     Y = [y]

                    #     while total_runtime <= constants.SOLVER_TIME_LIMIT:
                    #         # Compile diagram
                    #         diagram = DecisionDiagram()
                    #         diagram_manager = DecisionDiagramManager()
                    #         diagram = diagram_manager.compile_diagram(
                    #             diagram, instance, 
                    #             compilation_method=compilation_method, max_width=max_width, 
                    #             ordering_heuristic=ordering_heuristic, Y=Y
                    #         )
                    #         diagram.compilation_method = compilation_method

                    #         # Solve reformulation
                    #         result, solution = algorithms_manager.run_DD_reformulation(
                    #             instance, diagram, time_limit=time_limit, incumbent={"x": x, "y": Y[-1]}
                    #         )
                    #         # Track best solution
                    #         if result["lower_bound"] > best_result["lower_bound"]:
                    #             best_result = result
                    #         # Follower problem
                    #         _, y = solve_follower_problem(instance, solution["x"])
                    #         _, y = solve_aux_problem(instance, solution["x"], instance.d @ y)
                    #         # Collect solution
                    #         if y in Y:
                    #             break
                    #         Y.append(y)
                    #         x = solution["x"]
                    #         # Update times
                    #         total_runtime += result["total_runtime"]
                    #         compilation_runtime += result["compilation_runtime"]
                    #         reduce_algorithm_runtime += result["reduce_algorithm_runtime"]
                    #         model_runtime += result["model_runtime"]
                    #         time_limit -= result["total_runtime"]
                    #         # Track lower bound
                    #         lb_tracking.append((result["lower_bound"], total_runtime))

                    #     # Update final result
                    #     best_result["total_runtime"] = total_runtime
                    #     best_result["compilation_runtime"] = compilation_runtime
                    #     best_result["model_runtime"] = model_runtime
                    #     best_result["time_limit"] = constants.SOLVER_TIME_LIMIT
                    #     best_result["lower_bound_tracking"] = lb_tracking
                    #     best_result["width"] = result["width"]
                    #     best_result["initial_width"] = result["initial_width"]
                    #     best_result["num_vars"] = result["num_vars"]
                    #     best_result["num_constrs"] = result["num_constrs"]
                    #     best_result["HPR"] = HPR_value
                    #     best_result["Y_length"] = len(Y)
                    #     best_result["num_nodes"] = diagram.node_count + 2
                    #     best_result["num_arcs"] = diagram.arc_count
                    #     result = best_result
                    # else:

                    ## One-time compilation approach
                    result = algorithms_manager.one_time_compilation_approach(
                        instance, compilation_method, max_width, ordering_heuristic, constants.SOLVER_TIME_LIMIT
                    )

                    # Write results
                    name = "MW{}-VO{}".format(max_width, ordering_heuristic)
                    name = parser.write_results(result, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_name", "-i", type=str)
    args = parser.parse_args()

    # Testing
    args.instance_name = constants.INSTANCES or "miplib/stein2710"
    
    run(args)
