import os
import shutil

def mkdir(path, override=True):
    if os.path.exists(path) and os.path.isdir(path):
        if override: shutil.rmtree(path)
        else: return

    os.mkdir(path)

def write_modified_model_mps_file(model, instance, diagram):
    mkdir("results/models", override=False)
    mkdir("results/models/miplib", override=False)
    mkdir("results/models/miplib/DD{}".format(diagram.max_width), override=False)
    model.write("results/models/miplib/DD{}/{}.mps".format(diagram.max_width, instance.name.split("/")[1]))

def copy_aux_file(instance, diagram):
    shutil.copy("instances/{}.aux".format(instance.name), "results/models/miplib/DD{}/{}.aux".format(diagram.max_width, instance.name.split("/")[1]))