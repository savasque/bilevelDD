import logging, logzero
import argparse

import constants
from utils.parser import Parser

from algorithms.algorithms_manager import AlgorithmsManager

# LogLevel
logzero.loglevel(logging.getLevelName(constants.LOG_LEVEL))
logger = logzero.logger

def run(args):
    # Parser instantiation
    parser = Parser()

    # Load data
    instance = parser.build_instance(args.instance_name, args.problem_type)

    # Initiate AlgorithmsManager
    algorithms_manager = AlgorithmsManager(instance, args.num_threads, args.mip_solver, args.problem_setting)

    # Solve
    algorithms_manager.solve(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_name", type=str)
    parser.add_argument("--problem_type", type=str, default="general")  # general | bisp-kc
    parser.add_argument("--problem_setting", type=str, default="optimistic")  # optimistic | pessimistic
    parser.add_argument("--time_limit", type=int, default=3600)  # nonnegative integer (in seconds)
    parser.add_argument("--dd_max_width", type=int, default=0)  # integer >= -1 (-1 for exact DD)
    parser.add_argument("--dd_encoding", type=str, default="compact")  # compact | extended
    parser.add_argument("--dd_ordering_heuristic", type=str, default="lexicographic")  # lexicographic | 
    parser.add_argument("--dd_reduce_method", type=str, default="minmax_state")  # follower_cost | minmax_state | random
    parser.add_argument("--approach", type=str, default="iterative")  # iterative
    parser.add_argument("--mip_solver", type=str, default="gurobi")  # gurobi | cplex
    parser.add_argument("--num_threads", type=int, default=0)  # nonnegative integer (0 for all available threads)
    args = parser.parse_args()
  
    run(args)
