import numpy as np
import gurobipy as gp

from utils.utils import mkdir

np.random.seed(1)

class InstanceGenerator: 
    def generate_instance(self, instance_type, number_of_instances, params):
        path = "instances/custom"
        mkdir(path, override=False)
        path = "{}/{}".format(path, params["folder_name"])
        mkdir(path)

        for k in range(number_of_instances):
            # Constrs coeffs and RHS: leader's problem
            leader_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(0, 1, params["n_F"]))) for _ in range(params["m_L"])])
            leader_rhs = np.random.randint(0, 3 * params["n_L"], params["m_L"])
            # Constrs coeffs and RHS: follower's problem
            follower_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 11, params["n_F"]))) for _ in range(params["m_F"])])
            follower_rhs = np.random.randint(0, 3 * params["n_F"], params["m_F"])

            # ObjFunction: leader
            leader_obj = np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 0, params["n_F"])))
            # ObjFunction: follower
            follower_obj = np.array(np.random.randint(0, 11, params["n_F"]))

            # Aux file
            LL = list()
            LL.append("N {}".format(params["n_F"]))
            LL.append("M {}".format(params["m_F"]))
            for j in range(params["n_L"], params["n_L"] + params["n_F"]):
                LL.append("LC {}".format(j))
            for i in range(params["m_L"], params["m_L"] + params["m_F"]):
                LL.append("LR {}".format(i))
            for j in range(params["n_F"]):
                LL.append("LO {}".format(follower_obj[j]))
            LL.append("OS 1")
            with open("{}/instance-{}.aux".format(path, k), "w") as aux_file:
                aux_file.write("\n".join(LL))

            # MPS file
            model = gp.Model()
            x = model.addVars(range(params["n_L"]), vtype=gp.GRB.BINARY, name="x")
            y = model.addVars(range(params["n_F"]), vtype=gp.GRB.BINARY, name="y")

            obj = leader_obj @ (x.values() + y.values())
            model.addConstrs(leader_constrs[i] @ (x.values() + y.values()) <= leader_rhs[i] for i in range(params["m_L"])) 
            model.addConstrs(follower_constrs[i] @ (x.values() + y.values()) <= follower_rhs[i] for i in range(params["m_F"])) 
            model.setObjective(obj, sense=gp.GRB.MINIMIZE)
            
            model.write("{}/instance-{}.mps".format(path, k))