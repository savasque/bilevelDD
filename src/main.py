import logging, logzero
import argparse

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

    # Simulation
    if type(args.instance_name) != list:
        instances = [args.instance_name]
    else:
        instances = args.instance_name
    for instance_name in instances:
        for max_width in constants.MAX_WIDTH:
            for ordering_heuristic in constants.ORDERING_HEURISTIC:
                for compilation_method in constants.COMPILATION_METHOD: 
                    for approach in constants.APPROACH:
                        # Load data
                        instance = parser.build_instance(instance_name)

                        if approach == "one_time_compilation":
                            ## One-time compilation approach
                            result = algorithms_manager.one_time_compilation_approach(
                                instance, compilation_method, max_width, ordering_heuristic, constants.SOLVER_TIME_LIMIT
                            )
                        elif approach == "iterative":
                            ## Iterative compilation approach
                            result = algorithms_manager.iterative_compilation_approach(
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
    if not args.instance_name:
        args.instance_name = constants.INSTANCES
    
    run(args)
