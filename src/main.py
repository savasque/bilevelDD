import logging, logzero

from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from classes.decision_diagram_manager.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager

## Parameters
# General
LOG_LEVEL = "INFO"
# Compilation
COMPILATION = "restricted"
COMPILATION_METHOD = ["follower_leader"]
MAX_WIDTH = [128, 256, 512]
ORDERING_HEURISTIC = ["lhs_coeffs", "cost_leader", "cost_competitive", "leader_feasibility"]
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
    instances = ["40_3_25_1"] #["20_{}_25_1".format(i) for i in [1, 2, 3, 5]]
    results = list()
    for instance_name in instances:
        for max_width in MAX_WIDTH:
            for ordering_heuristic in ORDERING_HEURISTIC:
                for compilation_method in COMPILATION_METHOD:
                    # Load data
                    instance = parser.build_instance(instance_name) 
                    # Compile diagram
                    diagram = DecisionDiagram()
                    diagram_manager = DecisionDiagramManager()
                    diagram = diagram_manager.compile_diagram(
                        diagram, instance, compilation=COMPILATION, 
                        compilation_method=compilation_method, max_width=max_width, 
                        ordering_heuristic=ordering_heuristic
                    )
                    # Algorithm
                    result = algorithms_manager.run_DD_reformulation(instance, diagram, time_limit=SOLVER_TIME_LIMIT)
                    # Collect results
                    results.append(result)
    # Write results
    parser.write_results(results)

run()