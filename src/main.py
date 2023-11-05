import logging, logzero

from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager
from algorithms.utils.solve_HPR import run as solve_HPR
from algorithms.utils.solve_follower_problem import run as solve_follower_problem
from algorithms.utils.solve_aux_problem import run as solve_aux_problem
from algorithms.utils.collect_Y import run as collect_Y

## Parameters
# General
LOG_LEVEL = "INFO"
# Compilation
COMPILATION = "restricted"
COMPILATION_METHOD = ["follower_leader"] #["follower_leader", "leader_follower", "iterative", "collect_Y"]
MAX_WIDTH = [2500]
ORDERING_HEURISTIC = ["cost_competitive"] #["lhs_coeffs", "cost_leader", "cost_competitive", "leader_feasibility"]
# Solver
SOLVER_TIME_LIMIT = 3600

# LogLevel
logzero.loglevel(logging.getLevelName(LOG_LEVEL))
logger = logzero.logger

def run():
    # Classes instantiation
    parser = Parser()
    algorithms_manager = AlgorithmsManager()

    Y_tracker = dict()
    collect_Y_runtime = 0

    # Simulation
    instances = ["other/{}_{}_25_1".format(i, j) for i in [25, 50, 75] for j in [1, 2, 3]]
    # instances = ["other/75_1_25_1"]
    for instance_name in instances:
        for max_width in MAX_WIDTH:
            for ordering_heuristic in ORDERING_HEURISTIC:
                for compilation_method in COMPILATION_METHOD:
                    # Load data
                    instance = parser.build_instance(instance_name)
                    # Get HPR bound
                    HPR_value, HPR_solution = solve_HPR(instance)
                    _, y = solve_follower_problem(instance, HPR_solution["x"])
                    _, y = solve_aux_problem(instance, HPR_solution["x"], instance.d @ y)

                    if compilation_method == "iterative":
                        ## Iterative compilation approach
                        time_limit = int(SOLVER_TIME_LIMIT)
                        best_result = {"lower_bound": -float("inf")}
                        lb_tracking = list()
                        total_runtime = 0
                        compilation_runtime = 0
                        reduce_algorithm_runtime = 0
                        model_runtime = 0
                        # Collect solution
                        x = vars["x"]
                        Y = [y]

                        while total_runtime <= SOLVER_TIME_LIMIT:
                            # Compile diagram
                            diagram = DecisionDiagram()
                            diagram_manager = DecisionDiagramManager()
                            diagram = diagram_manager.compile_diagram(
                                diagram, instance, compilation=COMPILATION, 
                                compilation_method=compilation_method, max_width=max_width, 
                                ordering_heuristic=ordering_heuristic, Y=Y
                            )
                            diagram.compilation_method = compilation_method

                            # Solve reformulation
                            result, solution = algorithms_manager.run_DD_reformulation(
                                instance, diagram, time_limit=time_limit, incumbent={"x": x, "y": Y[-1]}
                            )
                            # Track best solution
                            if result["lower_bound"] > best_result["lower_bound"]:
                                best_result = result
                            # Follower problem
                            _, y = solve_follower_problem(instance, solution["x"])
                            _, y = solve_aux_problem(instance, solution["x"], instance.d @ y)
                            # Collect solution
                            if y in Y:
                                break
                            Y.append(y)
                            x = solution["x"]
                            # Update times
                            total_runtime += result["total_runtime"]
                            compilation_runtime += result["compilation_runtime"]
                            reduce_algorithm_runtime += result["reduce_algorithm_runtime"]
                            model_runtime += result["model_runtime"]
                            time_limit -= result["total_runtime"]
                            # Track lower bound
                            lb_tracking.append((result["lower_bound"], total_runtime))

                        # Update final result
                        best_result["total_runtime"] = total_runtime
                        best_result["compilation_runtime"] = compilation_runtime
                        best_result["model_runtime"] = model_runtime
                        best_result["time_limit"] = SOLVER_TIME_LIMIT
                        best_result["lower_bound_tracking"] = lb_tracking
                        best_result["width"] = result["width"]
                        best_result["initial_width"] = result["initial_width"]
                        best_result["num_vars"] = result["num_vars"]
                        best_result["num_constrs"] = result["num_constrs"]
                        best_result["HPR"] = HPR_value
                        best_result["Y_length"] = len(Y)
                        result = best_result
                    else:
                        ## One-time compilation approach
                        diagram = DecisionDiagram()
                        diagram_manager = DecisionDiagramManager()
                        Y = None
                        # Build set Y
                        if compilation_method == "collect_Y":
                            if instance not in Y_tracker:
                                # Collect y's
                                Y_length = 1000
                                Y, collect_Y_runtime = collect_Y(instance, num_solutions=Y_length)
                                Y_tracker[instance] = {"Y": Y, "runtime": collect_Y_runtime}
                            else:
                                Y = Y_tracker[instance]["Y"]
                                collect_Y_runtime = Y_tracker[instance]["runtime"]
                        # Compile diagram
                        diagram = diagram_manager.compile_diagram(
                            diagram, instance, compilation=COMPILATION, 
                            compilation_method=compilation_method, max_width=max_width, 
                            ordering_heuristic=ordering_heuristic, 
                            Y=None if compilation_method != "collect_Y" else Y
                        )
                        diagram.compilation_method = compilation_method
                        # Solve reformulation
                        result, solution = algorithms_manager.run_DD_reformulation(
                            instance, diagram, time_limit=SOLVER_TIME_LIMIT - collect_Y_runtime,
                            incumbent=None if compilation_method != "collect_Y" else {"x": HPR_solution["x"], "y": y}
                        )
                        # Update final result
                        result["HPR"] = HPR_value
                        result["Y_length"] = None if not Y else len(Y)
                        result["Y_runtime"] = collect_Y_runtime
                        result["total_runtime"] += collect_Y_runtime

                    # Write results
                    name = "w{}-O{}".format(max_width, ordering_heuristic) if not Y else "w{}-O{}-Y{}".format(max_width, ordering_heuristic, len(Y))
                    name = parser.write_results(result, name)

run()