import logging, logzero

from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager
from algorithms.utils.solve_HPR import run as solve_HPR
from algorithms.utils.solve_follower_problem import run as solve_follower_problem
from algorithms.utils.solve_aux_problem import run as solve_aux_problem

## Parameters
# General
LOG_LEVEL = "INFO"
# Compilation
COMPILATION = "restricted"
COMPILATION_METHOD = ["iterative"] #["follower_leader", "leader_follower", "iterative"]
MAX_WIDTH = [10000]
ORDERING_HEURISTIC = ["leader_feasibility"] #["lhs_coeffs", "cost_leader", "cost_competitive", "leader_feasibility"]
# Solver
SOLVER_TIME_LIMIT = 1800

# LogLevel
logzero.loglevel(logging.getLevelName(LOG_LEVEL))
logger = logzero.logger

def run():
    # Classes instantiation
    parser = Parser()
    algorithms_manager = AlgorithmsManager()

    # Simulation
    instances = ["30_3_25_1"] #["{}_{}_25_1".format(vars, constrs) for vars in [30] for constrs in [1, 2, 3, 5]]
    results = list()
    for instance_name in instances:
        for max_width in MAX_WIDTH:
            for ordering_heuristic in ORDERING_HEURISTIC:
                for compilation_method in COMPILATION_METHOD:
                    # Load data
                    instance = parser.build_instance(instance_name) 

                    if compilation_method == "iterative":
                        time_limit = int(SOLVER_TIME_LIMIT)
                        best_result = {"lower_bound": -float("inf")}
                        
                        # Initialize set of y's for Y-compilation
                        lb_tracking = list()
                        runtime = 0
                        _, vars = solve_HPR(instance, obj="leader", sense="min")
                        _, y = solve_follower_problem(instance, vars["x"])
                        _, y = solve_aux_problem(instance, vars["x"], instance.d @ y)
                        x = vars["x"]
                        Y = [y]
                        while runtime <= SOLVER_TIME_LIMIT:
                            # Compile diagram
                            diagram = DecisionDiagram()
                            diagram_manager = DecisionDiagramManager()
                            diagram = diagram_manager.compile_diagram(
                                diagram, instance, compilation=COMPILATION, 
                                compilation_method=compilation_method, max_width=max_width, 
                                ordering_heuristic=ordering_heuristic, Y=Y
                            )
                            # Algorithm
                            result, solution = algorithms_manager.run_DD_reformulation(instance, diagram, time_limit=time_limit, incumbent={"x": x, "y": Y[-1]})
                            
                            # Track best solution
                            if result["lower_bound"] > best_result["lower_bound"]:
                                best_result = result

                            # Follower problem
                            lb_tracking.append(result["lower_bound"])
                            _, y = solve_follower_problem(instance, solution["x"])
                            _, y = solve_aux_problem(instance, solution["x"], instance.d @ y)

                            # Collect y
                            if y in Y:
                                break
                            Y.append(y)
                            x = solution["x"]

                            #Update time
                            runtime += result["total_runtime"]
                            time_limit -= result["total_runtime"]

                        # Update final result
                        result = best_result
                        result["time_limit"] = SOLVER_TIME_LIMIT
                        result["lower_bound_tracking"] = lb_tracking
                    else:
                        # Compile diagram
                        diagram = DecisionDiagram()
                        diagram_manager = DecisionDiagramManager()
                        diagram = diagram_manager.compile_diagram(
                            diagram, instance, compilation=COMPILATION, 
                            compilation_method=compilation_method, max_width=max_width, 
                            ordering_heuristic=ordering_heuristic
                        )

                        # Algorithm
                        result, solution = algorithms_manager.run_DD_reformulation(instance, diagram, time_limit=SOLVER_TIME_LIMIT)

                    # Collect results
                    results.append(result)
    # Write results
    parser.write_results(results)

run()