import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # To ignore Pandas warnings

import numpy as np
import pandas as pd
import logzero
from time import time
import json
import matplotlib.pyplot as plt
import networkx as nx
from textwrap import dedent

import gurobipy as gp

from classes.instance import Instance


class Parser:
    def __init__(self):
        self.logger = logzero.logger

    def load_mps_file(self, file_name):
        file = "{}.mps".format(file_name)
        model = gp.read(file)
        data = {
            "constrs": np.array(model.getA().todense()).astype(int),
            "sense": model.sense,
            "rhs": np.array(model.getAttr("RHS", model.getConstrs())).astype(int),
            "obj": np.array(model.getAttr("Obj", model.getVars())).astype(int),
        }

        return data

    def load_aux_file(self, file_name):
        aux_file = "{}.aux".format(file_name)
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
    
    def load_json_file(self, file_name):
        file = "{}.json".format(file_name)
        with open(file, "r") as file:
            data = json.load(file)

        return data

    def build_instance(self, file_name, problem_type):
        t0 = time()
        mps_file = self.load_mps_file(file_name)
        aux_file = self.load_aux_file(file_name)
        json_file = dict() if problem_type != "bisp-kc" else self.load_json_file(file_name)

        Lrows = [i for i in range(len(mps_file["constrs"])) if i not in aux_file["LR"]]
        Frows = aux_file["LR"]
        Lcols = [i for i in range(len(mps_file["obj"])) if i not in aux_file["LC"]]
        Fcols = aux_file["LC"]

        A = np.array([]) if not Lrows else mps_file["constrs"][Lrows[0]:Lrows[-1] + 1, Lcols[0]:Lcols[-1] + 1]
        B = np.array([]) if not Lrows else mps_file["constrs"][Lrows[0]:Lrows[-1] + 1, Fcols[0]:Fcols[-1] + 1]
        C = mps_file["constrs"][np.ix_(Frows, Lcols)]
        D = mps_file["constrs"][np.ix_(Frows, Fcols)]
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

        # Update leader and follower rows
        Lrows = [i for i in range(A.shape[0])]
        Frows = [i if not Lrows else Lrows[-1] + 1 + i for i in range(C.shape[0])]

        # Build graph for BISP-KC instances
        graph = None if problem_type != "bisp-kc" else nx.node_link_graph(json_file["nx_data"])

        data = {
            "id": file_name,
            "problem_type": problem_type,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "a": a,
            "b": b,
            "cL": mps_file["obj"][Lcols],
            "cF": mps_file["obj"][Fcols],
            "d": np.array([i for i in aux_file["LO"]]),
            "nL": len(Lcols),
            "mL": len(Lrows),
            "nF": len(Fcols),
            "mF": len(Frows),
            "graph": graph,
            "p": json_file.get("p")
        }

        self.logger.info(dedent(
            """
            Instance {} succesfully loaded:
            Problem type        = {}
            Leader columns      = {}
            Leader rows         = {}
            Follower columns    = {}
            Follower rows       = {}"""
        ).format(
            file_name, problem_type, len(Lcols), len(Lrows), len(Fcols), len(Frows)
        ).strip())

        instance = Instance(file_name, time() - t0, data)

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
        for j in range(len(Fcols)):
            if np.all([instance.D[i][j] <= 0 for i in range(len(Frows))]) and instance.d[j] < 0:
                instance.known_y_values[j] = 1
            elif np.all([instance.D[i][j] >= 0 for i in range(len(Frows))]) and instance.d[j] > 0:
                instance.known_y_values[j] = 0

        return instance

    def write_results(self, result, name):
        new_result = pd.DataFrame([result], index=["instance"])
        new_result["instance_id"] = new_result["instance"].apply(lambda x: x.split("/")[-1])
        new_result.set_index("instance", inplace=True)
        try:
            current_results = pd.read_excel("results/{}.xlsx".format(name))
            current_results.set_index("instance", inplace=True)
            current_results = pd.concat([current_results, new_result])
            current_results.to_excel("results/{}.xlsx".format(name))
        except:
            new_result.to_excel("results/{}.xlsx".format(name))
    
    def plot_graph(self, instance, result):
        edge_map = {value: key for key, value in instance.edge_map.items()}
        selected_egdes = [edge_map[idx] for idx, u in enumerate(result["vars"]["x"]) if u == 1]
        for e in instance.graph.edges:
            instance.graph.edges[e]["color"] = "black"
        for u, v in selected_egdes:
            instance.graph.add_edge(u, v, color="red")
        
        pos = nx.circular_layout(instance.graph)
        options = {
            "node_color": ["black" if u == 0 else "green" for u in result["opt_y"]], 
            "edge_color": [instance.graph.edges[e]["color"] for e in instance.graph.edges],
            "node_size": 800
        }
        labels = {v: "{}:{}".format(v, instance.d[v]) for v in range(len(instance.graph.nodes))}
        nx.draw_networkx_labels(instance.graph, pos, labels, font_size=10, font_color="whitesmoke")

        plt.title("{} - Fixed edges: {} - Budget: {}".format(instance.name, len(instance.graph.edges), instance.a[0]))
        nx.draw(instance.graph, pos, **options)
        plt.show()
