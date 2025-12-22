import os
import shutil

def mkdir(path, override=True):
    if os.path.exists(path) and os.path.isdir(path):
        if override: shutil.rmtree(path)
        else: return

    os.mkdir(path)