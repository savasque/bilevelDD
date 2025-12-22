## Parameters
# General
LOG_LEVEL = "DEBUG"

# Compilation
ORDERING_HEURISTIC = ["follower_cost"]  # ["lhs_coeffs", "leader_cost", "follower_cost", "leader_feasibility", "max_connected_degree"]
DISCARD_METHOD = ["follower_cost"]  #["follower_cost", "minmax_state", "random"]
SAMPLING_METHOD = "partitioning"

# Solver
SOLVER_TIME_LIMIT = 3600
BILEVEL_FREE_SET_SEP_TYPE = "SEP-1"

# Instances
MIPLIB_INSTANCES = [
    "miplib/stein2710", 
    "miplib/stein2750", 
    "miplib/stein2790", 
    "miplib/stein4510", 
    "miplib/stein4550", 
    "miplib/stein4590",
    "miplib/enigma10",
    "miplib/enigma50",
    "miplib/enigma90",
    "miplib/lseu10",
    "miplib/lseu50",
    "miplib/lseu90",
    "miplib/p003310",
    "miplib/p003350",
    "miplib/p003390",
    "miplib/p020110",
    "miplib/p020150",
    "miplib/p020190",
    "miplib/p028210",
    "miplib/p028250",
    "miplib/p028290",
    "miplib/p054810",
    "miplib/p054850",
    "miplib/p054890",
    "miplib/p275610",
    "miplib/p275650",
    "miplib/p275690",
    "miplib/l152lav10",
    "miplib/l152lav50",
    "miplib/l152lav90",
    "miplib/mod01010",
    "miplib/mod01050",
    "miplib/mod01090",
    "miplib/air0310",
    "miplib/air0350",
    "miplib/air0390",
    "miplib/air0410",
    "miplib/air0450",
    "miplib/air0490",
    "miplib/air0510",
    "miplib/air0550",
    "miplib/air0590",
    "miplib/fast050710",
    "miplib/fast050750",
    "miplib/fast050790",
    "miplib/cap600010",
    "miplib/cap600050",
    "miplib/cap600090",
    "miplib/mitre10",
    "miplib/mitre50",
    "miplib/mitre90",
    "miplib/nw0410",
    "miplib/nw0450",
    "miplib/nw0490",
    "miplib/seymour10",
    "miplib/seymour50",
    "miplib/seymour90",
    "miplib/harp210",
    "miplib/harp250",
    "miplib/harp290"
]

# Results
RESULT_TEMPLATE = {
    "instance"                      : "",
    "nL"                            : 0,
    "nF"                            : 0,
    "mL"                            : 0,
    "mF"                            : 0,
    "num_threads"                   : 0,
    "mip_solver"                    : "",
    "time_limit"                    : 0,
    "dd_encoding"                   : "",
    "dd_max_width"                  : 0,
    "dd_ordering_heuristic"         : "",
    "dd_reduce_method"              : "",
    "dd_width"                      : 0,
    "dd_nodes"                      : 0,
    "dd_arcs"                       : 0,
    "lower_bound"                   : -float("inf"),
    "upper_bound"                   : float("inf"),
    "gap"                           : float("inf"),
    "opt"                           : 0,
    "dd_compilation_runtime"        : 0,
    "dd_ordering_heuristic_runtime" : 0,
    "model_build_runtime"           : 0,
    "model_solve_runtime"           : 0,
    "total_runtime"                 : 0,
    "num_vars"                      : 0,
    "num_constrs"                   : 0,
    "BB_node_count"                 : 0,
    "root_bound"                    : -float("inf"),
    "HPR_bound"                     : float("inf"),
    "HPR_runtime"                   : 0
}