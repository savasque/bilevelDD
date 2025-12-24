import numpy as np
import json
import gurobipy as gp

from utils.utils import mkdir

np.random.seed(1)

class InstanceGenerator: 
    def generate_instance(self, instance_type, number_of_instances, params):
        path = "instances/custom"
        mkdir(path, override=False)
        path = "{}/{}".format(path, params["folder_name"])
        mkdir(path)

        # Create instances
        for k in range(number_of_instances):
            # General problems instances
            if instance_type == "uniform":
                # Constrs coeffs and RHS: leader's problem
                leader_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 11, params["n_F"]))) for _ in range(params["m_L"])])
                leader_rhs = .25 * leader_constrs.sum(axis=1).round()
                # Constrs coeffs and RHS: follower's problem
                follower_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 11, params["n_F"]))) for _ in range(params["m_F"])])
                follower_rhs = (.25 * follower_constrs.sum(axis=1)).round()

                # ObjFunction: leader
                leader_obj = np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 0, params["n_F"])))
                # ObjFunction: follower
                follower_obj = np.array(np.random.randint(0, 11, params["n_F"]))

            elif instance_type == "sparse_leader":
                # Constrs coeffs and RHS: leader's problem
                leader_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 11, params["n_F"]))) for _ in range(params["m_L"])])
                leader_rhs = .25 * leader_constrs.sum(axis=1).round()
                # Constrs coeffs and RHS: follower's problem
                follower_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]) * [0 if i <= .8 else 1 for i in np.random.random(params["n_L"])], np.random.randint(-10, 11, params["n_F"]))) for _ in range(params["m_F"])])
                follower_rhs = (.25 * follower_constrs.sum(axis=1)).round()

                # ObjFunction: leader
                leader_obj = np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 0, params["n_F"])))
                # ObjFunction: follower
                follower_obj = np.array(np.random.randint(0, 11, params["n_F"]))
            
            elif instance_type == "weak_leader":
                # Constrs coeffs and RHS: leader's problem
                leader_constrs = np.array([np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 11, params["n_F"]))) for _ in range(params["m_L"])])
                leader_rhs = .25 * leader_constrs.sum(axis=1).round()
                # Constrs coeffs and RHS: follower's problem
                follower_constrs = np.array([np.concatenate(((.01 * np.random.randint(-100, 101, params["n_L"])).round(), np.random.randint(-100, 101, params["n_F"]))) for _ in range(params["m_F"])])
                follower_rhs = (.25 * follower_constrs.sum(axis=1)).round()

                # ObjFunction: leader
                leader_obj = np.concatenate((np.random.randint(-10, 11, params["n_L"]), np.random.randint(-10, 0, params["n_F"])))
                # ObjFunction: follower
                follower_obj = np.array(np.random.randint(0, 11, params["n_F"]))
            
            ## Write instance files
            if instance_type in ["uniform", "sparse_leader", "weak_leader"]:
                # Aux file args
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

                # MPS file args
                model = gp.Model()
                x = model.addVars(range(params["n_L"]), vtype=gp.GRB.BINARY, name="x")
                y = model.addVars(range(params["n_F"]), vtype=gp.GRB.BINARY, name="y")

                obj = leader_obj @ (x.values() + y.values())
                model.addConstrs(leader_constrs[i] @ (x.values() + y.values()) <= leader_rhs[i] for i in range(params["m_L"])) 
                model.addConstrs(follower_constrs[i] @ (x.values() + y.values()) <= follower_rhs[i] for i in range(params["m_F"])) 
                model.setObjective(obj, sense=gp.GRB.MINIMIZE)

            # Write MPS file
            with open("{}/instance-{}.aux".format(path, k), "w") as aux_file:
                aux_file.write("\n".join(LL))
            # Write aux file
            model.write("{}/instance-{}.mps".format(path, k))