from time import time
import numpy as np
from collections import deque

import gurobipy as gp


class Sampler:
    def __init__(self, logger, sampling_method):
        self.logger = logger
        self.sample = {
            "pooling": self.sample_by_pooling,
            "partitioning": self.sample_by_partitioning
        }.get(sampling_method, "pooling")

    def sample_by_pooling(self, instance, num_solutions):
        self.logger.info("Sampling y solutions by pooling")
        t0 = time()
        X = dict()
        Y = dict()

        while len(Y) < num_solutions and time() - t0 <= 60:
            _, new_xs, new_ys = self.solve_follower_HPR(instance, X, Y, num_solutions=num_solutions - len(Y), obj="leader")
            if not new_ys:
                break
            for x in new_xs:
                X[str(x)] = x
            for y in new_ys:
                Y[str(y)] = y
            print(len(Y))
        
        X = list(X.values())
        Y = list(Y.values())

        return Y[:num_solutions], round(time() - t0)

    def hamming_distance(self, v, v_2):
        return gp.quicksum(v[j] for j in range(len(v)) if v_2[j] < .5) + gp.quicksum(1 - v[j] for j in range(len(v)) if v_2[j] > .5)

    def solve_follower_HPR(self, instance, forbidden_X, forbidden_Y, num_solutions, obj):
        model = gp.Model()
        model.Params.OutputFlag = 0

        model.Params.PoolSearchMode = 2  # Extra effort to find alternative optimal solutions
        model.Params.PoolSolutions = num_solutions

        x = model.addVars(instance.Lcols, vtype=gp.GRB.BINARY, name="x")
        y = model.addVars(instance.Fcols, vtype=gp.GRB.BINARY, name="y")
        t = model.addVar(lb=-gp.GRB.INFINITY, name="t")
        tau = model.addVars(instance.Frows, lb=-gp.GRB.INFINITY, name="tau")

        # HPR constrs
        model.addConstrs(instance.A[i] @ x.values() + instance.B[i] @ y.values() <= instance.a[i] for i in range(instance.Lrows))
        model.addConstrs(instance.C[i] @ x.values() + instance.D[i] @ y.values() <= instance.b[i] for i in range(instance.Frows))

        # Avoid repeating solutions
        for x_2 in forbidden_X.values():
            model.addConstr(self.hamming_distance(x, x_2) >= 1)
        for y_2 in forbidden_Y.values():
            model.addConstr(self.hamming_distance(y, y_2) >= 1)

        # t and tau definitions
        model.addConstrs(tau[i] >= instance.D[i] @ y.values() for i in range(instance.Frows))
        model.addConstrs(t >= tau[i] for i in range(instance.Frows))

        # Get known optimal y-values (Fischetti et al, 2017)
        known_y_values = dict()
        for j in range(instance.Fcols):
            if np.all([instance.D[i][j] <= 0 for i in range(instance.Frows)]) and instance.d[j] < 0:
                known_y_values[j] = 1
            elif np.all([instance.D[i][j] >= 0 for i in range(instance.Frows)]) and instance.d[j] > 0:
                known_y_values[j] = 0
        model.addConstrs(y[j] == val for j, val in known_y_values.items())

        # Objective function
        if obj == "leader":
            obj_func = instance.c_leader @ x.values() + instance.c_follower @ y.values()
        elif obj == "follower":
            obj_func = instance.c_leader @ x.values() + instance.d @ y.values()
        elif obj == "follower_only":
            obj_func = instance.d @ y.values()
        elif obj == "leader_feasibility":
            obj_func = gp.quicksum((instance.D[i] @ y.values()) for i in range(instance.Frows))
        elif obj == "other":
            obj_func = t + .00001 * gp.quicksum(tau[i] for i in range(instance.Frows)) + .00001 * instance.d @ y.values()
        model.setObjective(obj_func, sense=gp.GRB.MINIMIZE)

        model.optimize()

        if model.status == 2:
            X = dict()
            Y = dict()
            for i in range(model.SolCount):
                model.Params.solutionNumber = i
                solution = model.getAttr("Xn")
                x = [int(solution[j]) for j in range(instance.Lcols)]
                y = [int(solution[j]) for j in range(instance.Lcols, instance.Lcols + instance.Fcols)]
                X[str(x)] = x
                Y[str(y)] = y
            
            X = list(X.values())
            Y = list(Y.values())
            
            return model.ObjVal, X, Y
        else:
            return None, list(), list()
        
    def sample_by_partitioning(self, instance, num_solutions):
        self.logger.info("Sampling y solutions by partitioning")
        t0 = time()
        Y = dict()

        Lcols = instance.Lcols
        Fcols = instance.Fcols
        Frows = instance.Frows
        A = instance.A
        B = instance.B
        C = instance.C
        D = instance.D
        a = instance.a
        b = instance.b
        d = instance.d

        model = gp.Model()
        model.Params.OutputFlag = 0
        x = model.addMVar(Lcols, vtype=gp.GRB.BINARY, name="x")
        y = model.addMVar(Fcols, vtype=gp.GRB.BINARY, name="y")
        
        # HPR constrs
        if A.shape[0] != 0 and B.shape[0] != 0:
            model.addConstr((A @ x + B @ y <= a), name="LeaderHPR")
        elif A.shape[0] != 0:
            model.addConstr((A @ x <= a), name="LeaderHPR")
        elif B.shape[0] != 0:
            model.addConstr((B @ y <= a), name="LeaderHPR")
        model.addConstr((C @ x + D @ y <= b), name="FollowerHPR")

        # Get known optimal y-values (Fischetti et al, 2017)
        model.addConstrs(y[j] == val for j, val in instance.known_y_values.items())
        
        # Objective function
        obj_func = d @ y
        # obj_func = gp.quicksum(instance.D[i] @ list(y.values()) for i in range(instance.Frows))
        model.setObjective(obj_func, sense=gp.GRB.MINIMIZE)
        
        model.optimize()

        queue = deque([(model, y)])
        while queue and time() - t0 <= 60:
            if len(Y) == num_solutions:
                break

            model, y = queue.popleft()
            model.Params.TimeLimit = max(0, 60 - (time() - t0))
            model.optimize()

            if model.status == 2:
                y_value = np.array([i.X for i in y])

                # # Sanity check
                # if str(y) in Y:
                #     a = None

                Y[str(y_value)] = y_value
                self.logger.debug("Total sampled solutions: {} - Time elapsed: {} s".format(len(Y), round(time() - t0)))

                # Create new model with disjunction
                new_model = model.copy()
                x = np.array([new_model.getVarByName("x[{}]".format(j)) for j in range(Lcols)])
                new_y = np.array([new_model.getVarByName("y[{}]".format(j)) for j in range(Fcols)])
                z = new_model.addMVar(instance.Frows, vtype=gp.GRB.BINARY)

                M = 1e6
                new_model.addConstrs((C[i] @ x + D[i] @ y_value >= b[i] + 1 - M * (1 - z[i]) for i in range(Frows)))
                new_model.addConstr(np.ones(Frows) @ z >= 1)

                for y_2 in Y.values():
                    new_model.addConstr(self.hamming_distance(new_y, y_2) >= 1)
                
                queue.append((new_model, new_y))
        
        # # Sanity check
        # for y_value in Y.values():
        #     model = gp.Model()
        #     model.Params.OutputFlag = 0
        #     x = model.addVars(instance.Lcols, vtype=gp.GRB.BINARY, name="x")
        #     # HPR constrs
        #     model.addConstrs((instance.C[i] @ x.values() + instance.D[i] @ y_value <= instance.b[i] for i in range(instance.Frows)), name="follower_constrs")
        #     # Get known optimal y-values (Fischetti et al, 2017)
        #     model.addConstrs(y[j] == val for j, val in instance.known_y_values.items())
        #     # Objective function
        #     obj_func = 0
        #     model.setObjective(obj_func, sense=gp.GRB.MINIMIZE)
        #     model.optimize()
        #     if model.status != 2:
        #         a = None

        Y = list(Y.values())

        self.logger.debug("Finishing sampling -> Time elapsed: {} s".format(round(time() - t0)))
        
        return Y, time() - t0
