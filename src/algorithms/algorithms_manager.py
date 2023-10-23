import logzero
import numpy as np

from .formulations.DD_reformulation import get_model

class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def run_DD_reformulation(self, instance, diagram, time_limit):
        self.logger.info("Solving DD refomulation. Time limit: {} sec".format(time_limit))
        model, vars = get_model(instance, diagram, time_limit)
        model.optimize()
        self.logger.info("DD reformulation solved succesfully. Time elapsed: {} sec.".format(model.runtime))
        self.logger.debug("Solving DD refomulation relaxation.".format(time_limit))
        relaxed_model = model.relax()
        relaxed_model.Params.OutputFlag = 0
        relaxed_model.optimize()
        self.logger.debug("DD reformulation relaxation solved succesfully. Time elapsed: {} sec".format(relaxed_model.runtime))
        results = self.format_result(model, vars, approach="DD_reformulation", relaxed_model=relaxed_model)
        results["instance"] = instance.name
        results["width"] = diagram.width
        self.logger.debug("Results: {}".format(results))

        return results
    
    def run_restricted_DD_algorithm(self, instance, diagram, time_limit):
        return None

    def format_result(self, model, vars, approach, relaxed_model=None):
        data = {
            "approach": approach,
            "objval": model.ObjVal,
            "relaxation_objval": relaxed_model.ObjVal,
            "continuous_optimality_gap": None,
            "runtime": model.runtime,
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs
        }
        if relaxed_model and model.ObjVal != 0:
            data["continuous_optimality_gap"] = (model.ObjVal - relaxed_model.ObjVal) / abs(model.ObjVal)

        return data