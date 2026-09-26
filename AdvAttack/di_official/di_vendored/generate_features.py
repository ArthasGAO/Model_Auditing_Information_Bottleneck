"""
VENDORED from: https://github.com/cleverhans-lab/dataset-inference
File: src/generate_features.py, commit 24baf46 (2022-10-10).
Original authors: Maini, Yaghini, Papernot (ICLR 2021).

Only `get_random_label_only` (the Blind Walk / `--feature_type rand` feature
extractor used by the black-box protocol) is kept. Its body is copied
verbatim. Modifications from the original file:

  1. The other feature extractors (pgd / topgd / mingd), `feature_extractor`,
     `get_student_teacher` and the CLI `__main__` block are not included: they
     build WideResNets and read checkpoints from the authors' disk layout,
     which this framework replaces with its own model objects.
  2. The original top-level imports `from funcs import *`, `from attacks import *`,
     `from models import *`, `from train import epoch_test`, `import params`
     are replaced by one relative import of the four symbols the function
     actually uses (`rand_steps`, `norms_*_squeezed`, all from the unmodified
     attacks.py).
  3. `device` is a module-level name that the original file only assigns
     inside `if __name__ == "__main__"`. It is declared here as `None` and
     injected by the adapter (`generate_features.device = torch.device(...)`)
     before the function is called, the same rebinding technique used for the
     ADV-TRA vendoring.

All lines of `get_random_label_only` are identical to the released repository.
"""
import time

import numpy as np
import torch

from .attacks import rand_steps, norms_linf_squeezed, norms_l1_squeezed, norms_l2_squeezed

# Injected by the adapter (see module docstring, item 3).
device = None


def get_random_label_only(args, loader, model, num_images = 1000):
    print("Getting random attacks")
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10): #5 random starts
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                targets = None
                delta = rand_steps(model, X, y, args, target = targets)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d
