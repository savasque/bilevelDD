import gurobipy as gp
import numpy as np
import scipy
import logging, logzero

from classes.instance import Instance


class Parser:
    def __init__(self):
        self.logger = logzero.logger
        if logzero.loglevel == logging.DEBUG:
            self.logfile("parser_logfile.log")

    def load_mps_file(self, file_name):
        file = "instances/{}.mps".format(file_name)
        model = gp.read(file)
        data = {
            "constrs": np.array(model.getA().todense()).astype(int),
            "rhs": np.array(model.getAttr("RHS", model.getConstrs())).astype(int),
            "obj": np.array(model.getAttr("Obj", model.getVars())).astype(int)
        }

        return data

    def load_aux_file(self, file_name):
        aux_file = "instances/{}.aux".format(file_name)
        with open(aux_file, "r") as aux_file:
            data = {
                "N": list(),
                "M": list(),
                "LC": list(),  # LL vars
                "LR": list(),  # LL constrs
                "LO": list(),  # LL ObjFunc coeffs
                "OS": list()  # opt sense
            }
            for i in aux_file.read().splitlines():
                key, value = i.split(" ")[:2]
                data[key].append(int(value))

        return data

    def build_instance(self, file_name):
        mps_file = self.load_mps_file(file_name)
        aux_file = self.load_aux_file(file_name)
        Lrows = [i for i in range(0, len(mps_file["constrs"]) - aux_file["M"][0])]
        Frows = [i for i in range(len(mps_file["constrs"]) - aux_file["M"][0], len(mps_file["constrs"]))]
        Lcols = [i for i in range(0, len(mps_file["obj"]) - aux_file["N"][0])]
        Fcols = [i for i in range(len(mps_file["obj"]) - aux_file["N"][0], len(mps_file["obj"]))]

        data = {
            "id": file_name,
            "A": mps_file["constrs"][Lrows[0]:Lrows[-1] + 1, Lcols[0]:Lcols[-1] + 1],
            "B": mps_file["constrs"][Lrows[0]:Lrows[-1] + 1, Fcols[0]:Fcols[-1] + 1],
            "C": mps_file["constrs"][Frows[0]:Frows[-1] + 1, Lcols[0]:Lcols[-1] + 1],
            "D": mps_file["constrs"][Frows[0]:Frows[-1] + 1, Fcols[0]:Fcols[-1] + 1],
            "a": mps_file["rhs"][Lrows[0]:Lrows[-1] + 1],
            "b": mps_file["rhs"][Frows[0]:Frows[-1] + 1],
            "c_leader": mps_file["obj"][Lcols[0]:Lcols[-1] + 1],
            "c_follower": mps_file["obj"][Fcols[0]:Fcols[-1] + 1],
            "d": np.array([i for i in aux_file["LO"]]),
        }

        self.logger.info("Instance succesfully loaded.")

        return Instance(file_name, data)

if __name__ == "__main__":
    import sys
    sys.path.append("/home/savasquez/research/bilevelDD/")
    parser = Parser()
    instance = parser.build_instance("20_1_25_1")