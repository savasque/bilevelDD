## Parameters
# General
LOG_LEVEL = "DEBUG"

# Compilation
COMPILATION_METHOD = ["follower_then_compressed_leader"]
APPROACH = ["one_time_compilation"]  # ["one_time_compilation", "iterative"]
MAX_WIDTH = [25]
ORDERING_HEURISTIC = ["follower_cost"]  # ["lhs_coeffs", "leader_cost", "follower_cost", "leader_feasibility", "max_connected_degree"]
DISCARD_METHOD = ["maxmin_slack"]  #["follower_cost", "minmax_state", "random"]
SAMPLING_LENGTH = 0
SAMPLING_METHOD = "pooling"

# Solver
SOLVER_TIME_LIMIT = 600

# Instances
INSTANCES = [
    # "miplib/stein2710", 
    # "miplib/stein2750", 
    # "miplib/stein2790", 
    # "miplib/stein4510", 
    # "miplib/stein4550", 
    # "miplib/stein4590",
    # "miplib/enigma10",
    # "miplib/enigma50",
    # "miplib/enigma90",
    # "miplib/lseu10",
    "miplib/lseu50",
    # "miplib/lseu90",
    # "miplib/p003310",
    # "miplib/p003350",
    # "miplib/p003390",
    # "miplib/p020110",
    # "miplib/p020150",
    # "miplib/p020190",
    # "miplib/p028210",
    # "miplib/p028250",
    # "miplib/p028290",
    # "miplib/p054810",
    # "miplib/p054850",
    # "miplib/p054890",
    # "miplib/p275610",
    # "miplib/p275650",
    # # "miplib/p275690",
    # "miplib/l152lav10",
    # "miplib/l152lav50",
    # "miplib/l152lav90",
    # "miplib/mod01010",
    # "miplib/mod01050",
    # # "miplib/mod01090",
]