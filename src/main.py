import logging, logzero

from utils.parser import Parser

from classes.decision_diagram import DecisionDiagram
from classes.decision_diagram_manager import DecisionDiagramManager

from algorithms.algorithms_manager import AlgorithmsManager

# Parameters
WIDTH = 128
SOLVER_TIME_LIMIT = 60
LOG_LEVEL = "INFO"

# Log level
logzero.loglevel(logging.getLevelName(LOG_LEVEL))
logger = logzero.logger

def run():
    parser = Parser()
    diagram_manager = DecisionDiagramManager()
    algorithms_manager = AlgorithmsManager()

    instances = ["10_2_25_{}".format(i) for i in range(1, 11)]
    results = list()
    for instance_name in instances:
        # Load data
        instance = parser.build_instance(instance_name) 
        # Compile diagram
        diagram = DecisionDiagram()
        diagram = diagram_manager.compile_diagram(diagram, instance, compilation="complete", max_width=WIDTH)
        # Algorithms
        results.append(algorithms_manager.run_DD_reformulation(instance, diagram, time_limit=SOLVER_TIME_LIMIT))
    # Write results
    parser.write_results(results)

run()