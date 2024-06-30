## Parameters
# General
LOG_LEVEL = "DEBUG"

# Compilation
ORDERING_HEURISTIC = ["follower_cost"]  # ["lhs_coeffs", "leader_cost", "follower_cost", "leader_feasibility", "max_connected_degree"]
DISCARD_METHOD = ["follower_cost"]  #["follower_cost", "minmax_state", "random"]
SAMPLING_LENGTH = 100
SAMPLING_METHOD = "partitioning"

# Solver
SOLVER_TIME_LIMIT = 3600
BILEVEL_FREE_SET_SEP_TYPE = "SEP-2"
