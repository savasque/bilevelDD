import logging, logzero

from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from classes.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager

## Parameters
# General
LOG_LEVEL = "DEBUG"
# Compilation
COMPILATION = "restricted"
MAX_WIDTH = [128]
ORDERING_HEURISTIC = ["lhs_coeffs"] #["lhs_coeffs", "cost_leader", "cost_competitive"]
# Solver
SOLVER_TIME_LIMIT = 600

# LogLevel
logzero.loglevel(logging.getLevelName(LOG_LEVEL))
logger = logzero.logger

def run():
    # Classes instantiation
    parser = Parser()
    diagram_manager = DecisionDiagramManager()
    algorithms_manager = AlgorithmsManager()

    # Simulation
    instances = ["20_2_25_1"] #["20_{}_25_1".format(i) for i in [1, 2, 3, 5]]
    results = list()
    for instance_name in instances:
        for max_width in MAX_WIDTH:
            for ordering_heuristic in ORDERING_HEURISTIC:
                # Load data
                instance = parser.build_instance(instance_name) 
                # Compile diagram
                diagram = DecisionDiagram()
                diagram = diagram_manager.compile_diagram(diagram, instance, compilation=COMPILATION, max_width=max_width, ordering_heuristic=ordering_heuristic)
                # Algorithm
                result = algorithms_manager.run_DD_reformulation(instance, diagram, time_limit=SOLVER_TIME_LIMIT)
                # Collect results
                results.append(result)
    # Write results
    parser.write_results(results)

run()