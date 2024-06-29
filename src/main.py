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

    file_name = "{}|w{}|{}".format(args.instance_name.split("/")[1], args.max_width, args.approach)

    for discard_method in constants.DISCARD_METHOD:
        for ordering_heuristic in constants.ORDERING_HEURISTIC: 
            # Load data
            instance = parser.build_instance(args.instance_name)

            ## One-time compilation approach
            result = algorithms_manager.one_time_compilation_approach(
                instance, args.max_width, ordering_heuristic, 
                discard_method, constants.SOLVER_TIME_LIMIT, args.approach
            )

            # Remove solutions
            del result["solution"]
            del result["follower_response"]

            # Write results
            if args.approach != "write_model":
                parser.write_results(result, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_name", "-i", type=str) 
    parser.add_argument("--max_width", "-w", type=int, default=0)
    parser.add_argument("--approach", "-a", type=str)  #lazy_cuts:no_good_cuts, lazy_cuts:INC, disjunctions, relaxation, write_model
    args = parser.parse_args()

    # Testing
    INSTANCES = [
        "miplib/stein2710", 
        "miplib/stein2750", 
        "miplib/stein2790", 
        "miplib/stein4510", 
        "miplib/stein4550", 
        "miplib/stein4590",
        "miplib/enigma10",
        "miplib/enigma50",
        "miplib/enigma90",
        "miplib/lseu10",
        "miplib/lseu50",
        "miplib/lseu90",
        "miplib/p003310",
        "miplib/p003350",
        "miplib/p003390",
        "miplib/p020110",
        "miplib/p020150",
        "miplib/p020190",
        "miplib/p028210",
        "miplib/p028250",
        "miplib/p028290",
        "miplib/p054810",
        "miplib/p054850",
        "miplib/p054890",
        "miplib/p275610",
        "miplib/p275650",
        "miplib/p275690",
        "miplib/l152lav10",
        "miplib/l152lav50",
        "miplib/l152lav90",
        "miplib/mod01010",
        "miplib/mod01050",
        "miplib/mod01090",
    ]

    if not args.max_width:
        args.max_width = 25
    if not args.approach:
        args.approach = "lazy_cuts:INC"
    if not args.instance_name:
        for instance in INSTANCES:
            args.instance_name = instance
            run(args)
    else:    
        run(args)
