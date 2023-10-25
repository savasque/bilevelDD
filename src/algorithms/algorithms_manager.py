import logzero

from .formulations.DD_reformulation import get_model


class AlgorithmsManager:
    def __init__(self):
        self.logger = logzero.logger
    
    def run_DD_reformulation(self, instance, diagram, time_limit):
        self.logger.info("Solving DD refomulation. Time limit: {} sec".format(time_limit))
        model, vars = get_model(instance, diagram, time_limit)
        model.optimize()
        self.logger.info("DD reformulation solved succesfully. Time elapsed: {} sec.".format(model.runtime))
        self.logger.debug("LB: {}, MIPGap: {}".format(model.objBound, model.MIPGap))
        self.logger.info("Solving DD refomulation relaxation.".format(time_limit))
        relaxed_model = model.relax()
        relaxed_model.Params.OutputFlag = 0
        relaxed_model.optimize()
        self.logger.info("DD reformulation relaxation solved succesfully. Time elapsed: {} sec".format(relaxed_model.runtime))
        results = self.format_result(model, diagram, approach="DD_reformulation", relaxed_model=relaxed_model)
        results["instance"] = instance.name
        results["width"] = diagram.width
        self.logger.debug("Results: {}".format(results))

        return results

    def format_result(self, model, diagram, approach, relaxed_model=None):
        data = {
            "approach": approach,
            "compilation": diagram.compilation,
            "compilation_method": diagram.compilation_method,
            "max_width": diagram.max_width,
            "ordering_heuristic": diagram.ordering_heuristic,
            "time_limit": model.Params.timeLimit,
            "objval": model.objVal,
            "lower_bound": model.objBound,
            "MIPGap": model.MIPGap,
            "relaxation_objval": relaxed_model.objVal,
            "runtime": model.runtime,
            "num_vars": model.numVars,
            "num_constrs": model.numConstrs
        }

        return data