from time import time
from collections import deque
from textwrap import dedent
import numpy as np

from constants import SAMPLING_METHOD

from .compilers.general_compiler import GenaralCompiler
from .compilers.bisp_kc_compiler_optimistic import BISPOptimisticCompiler

from algorithms.utils.sampler import Sampler


class DDCompiler:
    def __init__(self, logger):
        self.logger = logger

    def compile(self, instance, args):
        """
            This method compiles a DD, starting with the follower and continuing with a single compressed leader layer.
            
            Args: diagram (class DecisionDiagram), instance (class Instance), max_width (int), ordering_heuristic (str), [optional] Y (list)
            Returns: diagram (class DecisionDiagram)
        """

        if args["max_width"] == 0:
            return None
        else:
            self.logger.info(dedent(
                """
                Compiling diagram:
                Encoding            = {encoding}
                Max width           = {max_width}
                Ordering heuristic  = {ordering_heuristic}
                Reduce method       = {reduce_method}"""
            ).format(**args).strip())

            if instance.problem_type == "general":
                compiler = GenaralCompiler(self.logger)
            elif instance.problem_type == "bisp-kc":    
                compiler = BISPOptimisticCompiler(self.logger)

            return compiler.compile(instance, args)