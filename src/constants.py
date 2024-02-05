## Parameters
# General
LOG_LEVEL = "DEBUG"
# Compilation
COMPILATION_METHOD = ["follower_then_compressed_leader"] #["follower_then_leader", "leader_then_follower", "iterative", "collect_Y", "follower_then_compressed_leader"]
MAX_WIDTH = [50]
ORDERING_HEURISTIC = ["cost_competitive"] #["lhs_coeffs", "cost_leader", "cost_competitive", "leader_feasibility"]
# Solver
SOLVER_TIME_LIMIT = 3600