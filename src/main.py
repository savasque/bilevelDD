import logging, logzero

from utils.parser import Parser
from classes.decision_diagram import DecisionDiagram
from classes.decision_diagram_manager import DecisionDiagramManager

# Log level
logzero.loglevel(logging.DEBUG)

def run():
    # Load data
    parser = Parser()
    instance = parser.build_instance("20_1_25_1")  # Parse data``
    diagram = DecisionDiagram()
    diagram_manager = DecisionDiagramManager()

    diagram = diagram_manager.compile_diagram(diagram, instance, max_width=128)
    a = None

run()
