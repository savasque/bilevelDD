import logging, logzero
import argparse
from datetime import datetime

import constants
from utils.parser import Parser

from algorithms.algorithms_manager import AlgorithmsManager

# LogLevel
logzero.loglevel(logging.getLevelName(constants.LOG_LEVEL))
logger = logzero.logger

def run(args):
    # Classes instantiation
    parser = Parser()
    algorithms_manager = AlgorithmsManager()

    file_name = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Simulation
    if type(args.instance_name) != list:
        instances = [args.instance_name]
    else:
        instances = args.instance_name
    for instance_name in instances:
        for max_width in constants.MAX_WIDTH:
            for discard_method in constants.DISCARD_METHOD:
                for ordering_heuristic in constants.ORDERING_HEURISTIC:
                    for compilation_method in constants.COMPILATION_METHOD: 
                        for approach in constants.APPROACH:
                            # Load data
                            instance = parser.build_instance(instance_name)

                            if approach == "one_time_compilation":
                                ## One-time compilation approach
                                result = algorithms_manager.one_time_compilation_approach(
                                    instance, compilation_method, max_width, ordering_heuristic, 
                                    discard_method, constants.SOLVER_TIME_LIMIT
                                )
                            elif approach == "iterative":
                                ## Iterative compilation approach
                                result = algorithms_manager.iterative_compilation_approach(
                                    instance, compilation_method, max_width, ordering_heuristic, 
                                    discard_method, constants.SOLVER_TIME_LIMIT
                                )

                            # Remove solutions
                            del result["vars"]
                            del result["opt_y"]

                            # Write results
                            file_name = parser.write_results(result, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_name", "-i", type=str)
    args = parser.parse_args()

    # Testing
    if not args.instance_name:
        args.instance_name = constants.INSTANCES
    
    run(args)
