import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # To ignore Pandas warnings

import gurobipy as gp
import numpy as np
import pandas as pd
import logzero

from classes.instance import Instance


class Parser:
    def __init__(self):
        self.logger = logzero.logger

    def load_mps_file(self, file_name):
        file = "instances/{}.mps".format(file_name)
        model = gp.read(file)
        data = {
            "constrs": np.array(model.getA().todense()).astype(int),
            "sense": model.sense,
            "rhs": np.array(model.getAttr("RHS", model.getConstrs())).astype(int),
            "obj": np.array(model.getAttr("Obj", model.getVars())).astype(int),
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
                data[key].append(int(float(value)))

        return data

    def build_instance(self, file_name):
        mps_file = self.load_mps_file(file_name)
        aux_file = self.load_aux_file(file_name)
        Lrows = [i for i in range(len(mps_file["constrs"])) if i not in aux_file["LR"]]
        Frows = aux_file["LR"]
        Lcols = [i for i in range(len(mps_file["obj"])) if i not in aux_file["LC"]]
        Fcols = aux_file["LC"]

        A = np.array([]) if not Lrows else mps_file["constrs"][Lrows[0]:Lrows[-1] + 1, Lcols[0]:Lcols[-1] + 1]
        B = np.array([]) if not Lrows else mps_file["constrs"][Lrows[0]:Lrows[-1] + 1, Fcols[0]:Fcols[-1] + 1]
        C = mps_file["constrs"][Frows[0]:Frows[-1] + 1, Lcols[0]:Lcols[-1] + 1]
        D = mps_file["constrs"][Frows[0]:Frows[-1] + 1, Fcols[0]:Fcols[-1] + 1]
        a = np.array([]) if not Lrows else mps_file["rhs"][Lrows[0]:Lrows[-1] + 1]
        b = mps_file["rhs"][Frows[0]:Frows[-1] + 1]
        Lsense = np.array([]) if not Lrows else mps_file["sense"][Lrows[0]:Lrows[-1] + 1]
        Fsense = mps_file["sense"][Frows[0]:Frows[-1] + 1]

        # Turn all constrs sense into "<="
        for i, sense in enumerate(Lsense):
            if sense == ">":
                A[i] = -A[i]
                B[i] = -B[i]
                a[i] = -a[i]
            if sense == "=":
                A = np.vstack((A, -A[i]))
                B = np.vstack((B, -B[i]))
                a = np.append(a, -a[i])
        for i, sense in enumerate(Fsense):
            if sense == ">":
                C[i] = -C[i]
                D[i] = -D[i]
                b[i] = -b[i]
            if sense == "=":
                C = np.vstack((C, -C[i]))
                D = np.vstack((D, -D[i]))
                b = np.append(b, -b[i])

        # Update Lrows and Frows
        Lrows = [i for i in range(A.shape[0])]
        Frows = [i if not Lrows else Lrows[-1] + 1 + i for i in range(C.shape[0])]

        data = {
            "id": file_name,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "a": a,
            "b": b,
            "c_leader": mps_file["obj"][Lcols[0]:Lcols[-1] + 1],
            "c_follower": mps_file["obj"][Fcols[0]:Fcols[-1] + 1],
            "d": np.array([i for i in aux_file["LO"]]),
        }

        self.logger.info("Instance {} succesfully loaded. LCols: {}, LRows: {}, FCols: {}, FRows: {}".format(
            file_name,
            len(Lcols),
            len(Lrows),
            len(Fcols),
            len(Frows)
        ))

        instance = Instance(file_name, data)

        # Compute interaction in follower constrs
        for i in range(len(Frows)):
            if np.any([C[i][j] != 0 for j in range(len(Lcols))]):
                if np.any([D[i][j] != 0 for j in range(len(Fcols))]):
                    instance.interaction[i] = "both"
                else:
                    instance.interaction[i] = "leader"
            else:
                instance.interaction[i] = "follower"

        # Compute known values (Fischetti et al, 2017)
        for j in range(instance.Fcols):
            if np.all([instance.D[i][j] <= 0 for i in range(instance.Frows)]) and instance.d[j] < 0:
                instance.known_y_values[j] = 1
            elif np.all([instance.D[i][j] >= 0 for i in range(instance.Frows)]) and instance.d[j] > 0:
                instance.known_y_values[j] = 0

        return instance

    def write_results(self, result, name):
        new_result = pd.DataFrame([result], index=["instance"])
        new_result.set_index("instance", inplace=True)
        try:
            current_results = pd.read_excel("results/{}.xlsx".format(name))
            current_results.set_index("instance", inplace=True)
            current_results = pd.concat([current_results, new_result])
            current_results.to_excel("results/{}.xlsx".format(name))
        except:
            new_result.to_excel("results/{}.xlsx".format(name))

        return name
    
    def plot_graph(self, instance, result):
        import networkx as nx
        import matplotlib.pyplot as plt
        import json
        from ast import literal_eval

        with open("instances/{}.json".format(instance.name), "r") as file:
            data = json.load(file)
            edge_map = {literal_eval(key): value for key, value in data["edge_map"].items()}
        
        graph = nx.Graph()
        for u in range(len(result["vars"]["y"])):
            graph.add_node(u)
        fixed_edges = [(i, j) for i in range(len(graph.nodes) - 1) for j in range(i + 1, len(graph.nodes)) if (i, j) not in edge_map]
        for u, v in fixed_edges:
            graph.add_edge(u, v, color="k")
        edge_map = {value: key for key, value in edge_map.items()}
        selected_egdes = [edge_map[idx] for idx, u in enumerate(result["vars"]["x"]) if u == 1]
        for u, v in selected_egdes:
            graph.add_edge(u, v, color="r")
        
        pos = nx.circular_layout(graph)
        options = {
            "node_color": ["k" if u == 0 else "green" for u in result["vars"]["y"]], 
            "edge_color": [graph[u][v]["color"] for u, v in graph.edges]
        }
        plt.title("{} - Fixed edges: {} - Budget: {}".format(instance.name, data["fixed_edges"], data["leader_budget"]))
        nx.draw(graph, pos, **options)
        plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append("/home/savasquez/research/bilevelDD/")
    parser = Parser()
    instance = parser.build_instance("20_1_25_1")