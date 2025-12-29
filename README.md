# BilevelDD

Research code for decision-diagram-based reformulations and algorithms for binary bilevel optimization.

## Overview
- Core implementation lives in `src/`.
- Instance data lives in `instances/`.
<!-- - Results parsing utilities live in `parse_results.py`. -->

## Requirements
- Python (3.9+ recommended)
- Gurobi (and a valid license) for `gurobipy`
- CPLEX (and a valid license) for `docplex`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running
The main entry point is `src/main.py`.

Example:

```bash
python src/main.py --instance_name <NAME> --problem_type general --problem_setting optimistic
```

Options:
- `--problem_type`: `general` or `bisp-kc`
- `--problem_setting`: `optimistic` or `pessimistic`
- `--time_limit`: seconds (default 3600)
- `--dd_max_width`: integer, `-1` for exact DD
- `--dd_encoding`: `compact` or `extended`
- `--dd_ordering_heuristic`: `lexicographic`, `lhs_coeffs`, `follower_cost`, `leader_cost`, `max_connected_degree`
- `--dd_reduce_method`: `follower_cost` or `random`
- `--approach`: `iterative`
- `--mip_solver`: `gurobi` or `cplex`
- `--num_threads`: integer, `0` for all available threads

## Citation
If you use this code, please cite:

Vasquez, S., Lozano, L., and van Hoeve, W.-J. (2025).
A single-level reformulation of binary bilevel programs using decision diagrams.
Mathematical Programming. https://doi.org/10.1007/s10107-025-02294-1

BibTeX:

```bibtex
@article{vasquez2025single,
  title = {A single-level reformulation of binary bilevel programs using decision diagrams},
  author = {Vasquez, Sebastian and Lozano, Leonardo and van Hoeve, Willem-Jan},
  journal = {Mathematical Programming},
  year = {2025},
  doi = {10.1007/s10107-025-02294-1},
  url = {https://link.springer.com/article/10.1007/s10107-025-02294-1}
}
```

## Contact
For help, please contact Sebastian Vasquez at `savasque@andrew.cmu.edu`.
