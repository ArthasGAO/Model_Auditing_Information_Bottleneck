import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from util import process_yaml_file, process_experiment_setup, create_train_subset, evaluate2, prepare_group_subset, \
                 process_experiment_setup_deit, create_or_load_group_A, create_class_balanced_mix_train_test, MixedSplitDataset, \
                 load_best_checkpoint, create_or_load_subset_from_group

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    # Worker gets a different, but deterministic, seed derived from the main seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        # Raises error if a non-deterministic op is used; great for debugging reproducibility
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

# =====================================================
# 3. Main training logic
# =====================================================
def main_nega(seed, r, yaml_file_path): # this function is used to calculate the model from main_train_nega.py
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r,2)}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    # ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    #train_set = exp_setup["Dataset"].train_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI_new'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
    value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            value_xt_out, value_ty_out
                            ])
        

def main_nega_10000(seed, r, yaml_file_path): # this function is to extract a 10000 subset from training samples 
                                              # to align with the mixed dataset size.
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r,2)}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    # ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    #train_set = exp_setup["Dataset"].train_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    subset_A = create_or_load_subset_from_group(
                dataset=in_sample_set,
                group_A=group_A,
                save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                subset_size=10000,
                num_classes=exp_setup["NumClasses"],
                seed=42,
                force_rebuild=False
            )
    
    in_sample_subset = exp_setup["Dataset"].subset("train", subset_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI_v1'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name + f"_{10000}_probing"
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
    value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            value_xt_out, value_ty_out
                            ])
        


def main_deit(seed, r, yaml_file_path): # this function is used to calculate the model from main_train_nega.py
    print(device)

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup_deit(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    #ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Student Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI/DeiT/'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
    value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            value_xt_out, value_ty_out
                            ])
        

def main_nega_overlapping(seed, r, yaml_file_path): # this function is used to calculate the mixed probe dataset MI 
                                                    # based on different overlapping rate
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{1.0}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    # ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    #train_set = exp_setup["Dataset"].train_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    
    mixed, picked_train, picked_test = create_class_balanced_mix_train_test(
                                        train_dataset=in_sample_set,
                                        test_dataset=out_sample_set,
                                        train_in_indices=group_A,
                                        num_classes=exp_setup["NumClasses"],
                                        total_size=10000, # here we temporarily set the total size of mixed probe dataset as 10000
                                        frac_in_from_train=round(r,2),
                                        seed=42,
                                        return_parts=True,
                                    )
    mixed_subset = MixedSplitDataset(in_sample_set, out_sample_set, mixed, return_split=False)
    mixed_sample_loader = DataLoader(mixed_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name + f"_overlap_{round(r,2)}_size_{10000}"
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, mixed_sample_loader, device)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            0, 0 # in this case, no need to compute the out-sample MI
                            ])


def main_deit_overlapping(seed, r, yaml_file_path):
    print(device)

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup_deit(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{1.0}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    #ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Student Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    mixed, picked_train, picked_test = create_class_balanced_mix_train_test(
                                        train_dataset=in_sample_set,
                                        test_dataset=out_sample_set,
                                        train_in_indices=group_A,
                                        num_classes=exp_setup["NumClasses"],
                                        total_size=10000, # here we temporarily set the total size of mixed probe dataset as 10000
                                        frac_in_from_train=round(r,2),
                                        seed=42,
                                        return_parts=True,
                                    )
    mixed_subset = MixedSplitDataset(in_sample_set, out_sample_set, mixed, return_split=False)
    mixed_sample_loader = DataLoader(mixed_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name + f"_overlap_{round(r,2)}_size_{10000}"
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, mixed_sample_loader, device)
    

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            0, 0
                            ])
        


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/train_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))
    
    if not yaml_files:
            print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")
            
        for seed in range(42, 52): # entry point for main_nega
            for rate in np.linspace(0, 1, 11): #overlapping rate: [0, 0.1, 0.2, 0.3,...,0.9, 1]
                print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
                set_seed(seed)

                main_nega(seed, rate, yaml_path)

        '''for seed in range(42, 45): # entry point for main_nega
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            main_nega(seed, 1.0, yaml_path)'''

            #main_deit(seed, 1.0, yaml_path)

            #for frac in np.arange(0.0, 1.1, 0.1):
                #
                
        
        '''for seed in range(42, 52): # entry point for main_nega_10000
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            for frac in np.arange(0.0, 1.1, 0.1):
                main_nega_10000(seed, frac, yaml_path)'''
            
        '''for seed in range(42, 52): # entry point for main_nega
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            for frac in np.arange(0.1, 1.0, 0.1):
                #main_nega_overlapping(seed, frac, yaml_path)
                main_deit_overlapping(seed, frac, yaml_path)'''
