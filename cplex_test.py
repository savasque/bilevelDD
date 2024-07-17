from docplex.mp.model import Model

model = Model(log_output=True, float_precision=6)
x = model.binary_var_dict(1001, name="x")

model.solve()