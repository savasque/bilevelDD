import logging, logzero

from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from classes.decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager
from algorithms.utils.solve_HPR import run as solve_HPR
from algorithms.utils.solve_follower_problem import run as solve_follower_problem

## Parameters
# General
LOG_LEVEL = "DEBUG"
# Compilation
COMPILATION = "restricted"
COMPILATION_METHOD = ["leader_follower"] #["follower_leader", "leader_follower", "follower_leader_Y"]
MAX_WIDTH = [128]
ORDERING_HEURISTIC = ["cost_leader"] #["lhs_coeffs", "cost_leader", "cost_competitive", "leader_feasibility"]
# Solver
SOLVER_TIME_LIMIT = 600

# LogLevel
logzero.loglevel(logging.getLevelName(LOG_LEVEL))
logger = logzero.logger

def run():
    # Classes instantiation
    parser = Parser()
    algorithms_manager = AlgorithmsManager()

    # Simulation
    instances = ["20_5_25_1"] #["20_{}_25_1".format(i) for i in [1, 2, 3, 5]]
    results = list()
    for instance_name in instances:
        for max_width in MAX_WIDTH:
            for ordering_heuristic in ORDERING_HEURISTIC:
                for compilation_method in COMPILATION_METHOD:
                    # Load data
                    instance = parser.build_instance(instance_name) 

                    if compilation_method == "follower_leader_Y":
                        lb_tracking = list()
                        # Set of y's for Y-compilation
                        runtime = 0
                        _, vars = solve_HPR(instance, obj="leader", sense="min")
                        _, y = solve_follower_problem(instance, vars["x"])
                        Y = [y]
                        for i in range(3):
                            # Compile diagram
                            diagram = DecisionDiagram()
                            diagram_manager = DecisionDiagramManager()
                            diagram = diagram_manager.compile_diagram(
                                diagram, instance, compilation=COMPILATION, 
                                compilation_method=compilation_method, max_width=max_width, 
                                ordering_heuristic=ordering_heuristic, Y=Y
                            )
                            # Algorithm
                            result, solution = algorithms_manager.run_DD_reformulation(instance, diagram, time_limit=SOLVER_TIME_LIMIT)
                            # Collect y
                            _, y = solve_follower_problem(instance, solution["x"])
                            Y.append(y)

                            runtime += result["runtime"]
                            lb_tracking.append(result["lower_bound"])

                        result["runtime"] = runtime
                        result["lower_bound_tracking"] = lb_tracking
                    else:
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