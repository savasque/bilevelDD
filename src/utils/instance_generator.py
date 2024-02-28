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

            # Independent set instances
            elif instance_type == "independent_set":
                import networkx as nx
                p = np.random.uniform(low=0, high=.5)
                budget_ratio = np.random.uniform(low=0, high=.5)
                graph = nx.fast_gnp_random_graph(n=params["independent_set_params"]["n"], p=p, seed=k)
                elegible_edges = [(i, j) for i in range(len(graph.nodes) - 1) for j in range(i + 1, len(graph.nodes)) if (i, j) not in graph.edges]
                edge_map = {value: idx for idx, value in enumerate(elegible_edges)}
                
                # Leader problem
                leader_constrs = np.concatenate((np.ones(len(elegible_edges)), np.zeros(len(graph.nodes))))
                leader_rhs = np.array([round(budget_ratio * len(elegible_edges))])
                leader_obj = np.concatenate((np.zeros(len(elegible_edges)), np.ones(len(graph.nodes))))
                # Follower problem
                follower_constrs = list()
                follower_rhs = np.concatenate((2 * np.ones(len(elegible_edges)), np.ones(len(graph.edges))))
                for edge in elegible_edges:
                    var_index = edge_map[edge]
                    row = np.zeros(len(elegible_edges) + len(graph.nodes))
                    row[var_index] = 1
                    row[len(elegible_edges) + edge[0]] = 1
                    row[len(elegible_edges) + edge[1]] = 1
                    follower_constrs.append(row)
                for edge in graph.edges:
                    row = np.zeros(len(elegible_edges) + len(graph.nodes))
                    row[len(elegible_edges) + edge[0]] = 1
                    row[len(elegible_edges) + edge[1]] = 1
                    follower_constrs.append(row)
                follower_constrs = np.array(follower_constrs) 
                follower_obj = -np.ones(len(graph.nodes))
            
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
            
            elif instance_type == "independent_set":
                # Extra info file
                data = {"leader_budget": int(leader_rhs[0]), "fixed_edges": len(graph.edges), "edge_map": {str(key): value for key, value in edge_map.items()}}
                with open("{}/instance-{}.json".format(path, k), "w") as file:
                    json.dump(data, file)

                # Aux file args
                LL = list()
                LL.append("N {}".format(len(graph.nodes)))
                LL.append("M {}".format(len(follower_constrs)))
                for j in range(len(elegible_edges), len(elegible_edges) + len(graph.nodes)):
                    LL.append("LC {}".format(j))
                for i in range(1, 1 + len(follower_constrs)):
                    LL.append("LR {}".format(i))
                for j in range(len(graph.nodes)):
                    LL.append("LO {}".format(follower_obj[j]))
                LL.append("OS 1")

                # MPS file args
                model = gp.Model()
                x = model.addVars(range(len(elegible_edges)), vtype=gp.GRB.BINARY, name="x")
                y = model.addVars(range(len(graph.nodes)), vtype=gp.GRB.BINARY, name="y")

                obj = leader_obj @ (x.values() + y.values())
                model.addConstr(leader_constrs @ (x.values() + y.values()) <= leader_rhs) 
                model.addConstrs(follower_constrs[i] @ (x.values() + y.values()) <= follower_rhs[i] for i in range(len(follower_constrs))) 
                model.setObjective(obj, sense=gp.GRB.MINIMIZE)

            # Write MPS file
            with open("{}/instance-{}.aux".format(path, k), "w") as aux_file:
                aux_file.write("\n".join(LL))
            # Write aux file
            model.write("{}/instance-{}.mps".format(path, k))