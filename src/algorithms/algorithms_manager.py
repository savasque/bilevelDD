import logzero

from .formulations.DD_reformulation import get_model
from .utils.solve_follower_problem import run as solve_follower_problem
from .utils.solve_aux_problem import run as solve_aux_problem


class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def run_DD_reformulation(self, instance, diagram, time_limit, incumbent=dict()):
        # Solve DD reformulation
        self.logger.info("Solving DD refomulation. Time limit: {} sec".format(time_limit))
        model, vars = get_model(instance, diagram, time_limit, incumbent)
        model.optimize()
        self.logger.info("DD reformulation solved succesfully. Time elapsed: {} sec.".format(model.runtime))
        self.logger.debug("LB: {}, MIPGap: {}".format(model.objBound, model.MIPGap))
        
        # # Solve continuous relaxation
        # self.logger.info("Solving DD refomulation relaxation.".format(time_limit))
        # relaxed_model = model.relax()
        # relaxed_model.Params.OutputFlag = 0
        # relaxed_model.optimize()
        # self.logger.info("DD reformulation relaxation solved succesfully. Time elapsed: {} sec".format(relaxed_model.runtime))
        
        # Format results
        results = self.format_result(model, diagram, approach="DD_reformulation") #, relaxed_model=relaxed_model)
        results["instance"] = instance.name
        results["width"] = diagram.width
        results["initial_width"] = diagram.initial_width
        self.logger.debug("Results: {}".format(results))
        try:
            vars = {
                "x": [i.X for i in vars["x"].values()],
                "y": [i.X for i in vars["y"].values()],
                "w": {key: value.X for key, value in vars["w"].items()}
            }
        except:
            vars = dict()
        follower_value = solve_follower_problem(instance, vars["x"])[0]
        follower_response = solve_aux_problem(instance, vars["x"], follower_value)[1]
        results["upper_bound"] = instance.c_leader @ vars["x"] + instance.c_follower @ follower_response

        return results, vars

    def format_result(self, model, diagram, approach, relaxed_model=None):
        try:
            objval = model.objVal
        except:
            objval = None
        data = {
            "approach": approach,
            "compilation": diagram.compilation,
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "time_limit": model.Params.timeLimit,
            "objval": objval,
            "lower_bound": model.objBound,
            "MIPGap": model.MIPGap,
            "relaxation_objval": None if not relaxed_model else relaxed_model.objVal,
            "total_runtime": diagram.compilation_runtime + diagram.reduce_algorithm_runtime + model.runtime,
            "compilation_runtime": diagram.compilation_runtime,
            "reduce_algorithm_runtime": diagram.reduce_algorithm_runtime,
            "model_runtime": model.runtime,
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs
        }

        return data