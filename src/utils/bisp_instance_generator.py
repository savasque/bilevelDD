import numpy as np
import json
import gurobipy as gp
import networkx as nx

from utils.utils import mkdir

np.random.seed(1)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class InstanceGenerator: 
    def generate_instance(self, number_of_instances, params):
        instances_path = "instances/bisp-side-constr"
        mkdir(instances_path, override=False)

        # Create instances
        for nL in params["nL"]:
            for nF in params["nF"]:
                for s in range(number_of_instances):
                    c_leader = np.random.randint(-50, 51, size=nL)
                    c_follower = np.random.randint(-50, 51, size=nF)
                    r = np.random.randint(25, 51, size=nF) # np.ones(nF)
                    A = np.random.randint(-10, 11, size=nL)
                    B = np.random.randint(-10, 11, size=nF)
                    C_0 = np.random.randint(-10, 11, size=nL)
                    D_0 = np.random.randint(-10, 11, size=nF)
                    for p in params["p"]:
                        ## Graph
                        graph = nx.fast_gnp_random_graph(n=nF, p=p, seed=s)
                        ## Follower problem
                        C = np.vstack((C_0, np.zeros((len(graph.edges), nL))))
                        D = np.vstack((D_0, nx.incidence_matrix(graph).transpose().toarray()))

                        # Leader and follower RHS
                        for rhs_ratio in params["rhs_ratio"]:
                            a = np.floor(rhs_ratio * (nL + nF))
                            b = np.floor(rhs_ratio * (nL + nF))
                            b = np.insert(np.ones(len(graph.edges)), 0, b)

                            # Aux file args
                            LL = list()
                            LL.append("N {}".format(nF))
                            LL.append("M {}".format(len(C)))
                            for j in range(nL, nL + nF):
                                LL.append("LC {}".format(j))
                            for i in range(1, 1 + len(C)):
                                LL.append("LR {}".format(i))
                            for j in range(nF):
                                LL.append("LO {}".format(-r[j]))
                            LL.append("OS 1")

                            # MPS file args
                            model = gp.Model()
                            x = model.addMVar(nL, vtype=gp.GRB.BINARY, name="x")
                            y = model.addMVar(nF, vtype=gp.GRB.BINARY, name="y")

                            obj = c_leader @ x + c_follower @ y
                            model.addConstr(A @ x + B @ y <= a) 
                            model.addConstr(C @ x + D @ y <= b) 
                            model.setObjective(obj, sense=gp.GRB.MINIMIZE)

                            ## Write files
                            folder_name = "nL{}_nF{}_p{}_r{}".format(nL, nF, int(p * 100), int(rhs_ratio * 100))
                            path = "{}/{}".format(instances_path, folder_name)
                            mkdir(path, override=False)
                            
                            # Write MPS file
                            with open("{}/instance-{}.aux".format(path, s), "w") as aux_file:
                                aux_file.write("\n".join(LL))
                            
                            # Write aux file
                            model.write("{}/instance-{}.mps".format(path, s))
                            
                            # Extra info file
                            data = {
                                "nx_data": nx.node_link_data(graph), 
                                "p": p
                            }
                            with open("{}/instance-{}.json".format(path, s), "w") as file:
                                json.dump(data, file, cls=NpEncoder)

                            print("Instance created: nL={}, nF={}, p={}, rhs_ratio={}, seed={}".format(nL, nF, p, rhs_ratio, s))
    
    def generate_example_instance(self):
        instances_path = "instances/example"
        mkdir(instances_path, override=False)

        # Create instances
        c_leader = np.random.randint(-50, 51, size=4)
        c_follower = np.random.randint(-50, 51, size=6)
        r = np.random.randint(25, 51, size=6)
        A = np.random.randint(-10, 11, size=4)
        B = np.random.randint(-10, 11, size=6)
        C_0 = np.random.randint(-10, 11, size=4)
        D_0 = np.random.randint(-10, 11, size=6)

        ## Graph
        graph = nx.Graph(n=6)
        graph.add_edges_from([(0, 1), (0, 5), (1, 2), (1, 5), (2, 3), (2, 4), (4, 5)])

        ## Follower problem
        C = np.vstack((C_0, np.zeros((len(graph.edges), 4))))
        D = np.vstack((D_0, nx.incidence_matrix(graph).transpose().toarray()))

        # Leader and follower RHS
        a = np.floor(.1 * 10)
        b = np.floor(.1 * 10)
        b = np.insert(np.ones(len(graph.edges)), 0, b)

        # Aux file args
        LL = list()
        LL.append("N {}".format(6))
        LL.append("M {}".format(len(C)))
        for j in range(4, 10):
            LL.append("LC {}".format(j))
        for i in range(1, 1 + len(C)):
            LL.append("LR {}".format(i))
        for j in range(6):
            LL.append("LO {}".format(-r[j]))
        LL.append("OS 1")

        # MPS file args
        model = gp.Model()
        x = model.addMVar(4, vtype=gp.GRB.BINARY, name="x")
        y = model.addMVar(6, vtype=gp.GRB.BINARY, name="y")

        obj = c_leader @ x + c_follower @ y
        model.addConstr(A @ x + B @ y <= a) 
        model.addConstr(C @ x + D @ y <= b) 
        model.setObjective(obj, sense=gp.GRB.MINIMIZE)

        ## Write files
        folder_name = "example"
        path = "{}/{}".format(instances_path, folder_name)
        mkdir(path, override=False)
        
        # Write MPS file
        with open("{}/instance-0.aux".format(path), "w") as aux_file:
            aux_file.write("\n".join(LL))
        
        # Write aux file
        model.write("{}/instance-0.mps".format(path))
        
        # Extra info file
        data = {
            "nx_data": nx.node_link_data(graph)
        }
        with open("{}/instance-0.json".format(path), "w") as file:
            json.dump(data, file, cls=NpEncoder)