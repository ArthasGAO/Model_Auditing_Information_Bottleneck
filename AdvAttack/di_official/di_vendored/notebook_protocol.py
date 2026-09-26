"""
TRANSCRIBED from: https://github.com/cleverhans-lab/dataset-inference
Files: src/notebooks/CIFAR10_rand.ipynb (cells 3, 8-18, 20, 29) and
       src/notebooks/utils.py (generate_table), commit 24baf46 (2022-10-10).
Original authors: Maini, Yaghini, Papernot (ICLR 2021).

The notebook is the ONLY place where the official repository turns Blind Walk
features into a regressor and a p-value, and a notebook cannot be imported, so
each cell is wrapped in a function below. Cell bodies are copied verbatim.
The complete list of differences from the notebook:

  (a) Hard-coded `names` list and feature-file root become function arguments
      (the notebook fixes eight threat-model names; here they come from the
      YAML plan). The victim MUST still be called "teacher": the notebook uses
      trains["teacher"] for the standardization statistics and the regressor.
  (b) `Tensor.T` on the 3-D feature tensor -> `_reverse_dims` (identical
      semantics: reverse all dimensions; `.T` on N-D tensors is deprecated
      since torch 1.x and scheduled to raise). The equivalence is asserted in
      the smoke test.
  (c) `tqdm.notebook.tqdm` -> `tqdm.auto.tqdm` (no ipywidgets in a script).
  (d) `ipdb.set_trace()` on a negative p-value in generate_table -> raise
      (the notebook's own get_p already raises in the same situation).
  (e) The literal `1000` in `reshape(1000, f_num)` -> `num_images` argument
      (default 1000 = the notebook value; only changed for smoke tests).
  (f) Cell 20 (main loop over m): `results_df.to_hdf` needs PyTables and the
      stray `pbar.set_description(...)` refers to the training progress bar
      from cell 12; both lines are dropped and the DataFrame is returned.
  (g) `inference_stats` is ADDED: it returns the (p-value, mean difference)
      pair that `print_inference` only prints. `print_inference` is kept.

Everything else (standardization, .T.reshape layout, split_index rows,
regressor architecture / optimizer / epochs / loss, Welch test direction,
harmonic-mean protocol) is the notebook code.
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import combine_pvalues, ttest_ind_from_stats, ttest_ind
from functools import reduce
from scipy.stats import hmean
from tqdm.auto import tqdm


def _reverse_dims(t):
    """Same as `t.T` for an N-D tensor (reverse all dimensions); see (b)."""
    return t.permute(*range(t.ndim - 1, -1, -1))


# ---------------------------------------------------------------- cell 3
def set_notebook_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------- cell 8
def load_features(root, names, v_type="rand"):
    trains = {}
    tests = {}
    for name in names:
        trains[name] = (torch.load(f"{root}/model_{name}_normalized/train_{v_type}_vulnerability.pt"))
        tests[name] = (torch.load(f"{root}/model_{name}_normalized/test_{v_type}_vulnerability.pt"))
    mean_cifar = trains["teacher"].mean(dim = (0,1))
    std_cifar = trains["teacher"].std(dim = (0,1))
    return trains, tests, mean_cifar, std_cifar


# ---------------------------------------------------------------- cell 9
def normalize_and_flatten(trains, tests, names, mean_cifar, std_cifar, v_type="rand", num_images=1000):
    if v_type == "mingd":
        for name in names:
            trains[name] = trains[name].sort(dim = 1)[0]
            tests[name] = tests[name].sort(dim = 1)[0]

    for name in names:
        trains[name] = (trains[name]- mean_cifar)/std_cifar
        tests[name] = (tests[name]- mean_cifar)/std_cifar

    f_num = 30
    a_num=30

    trains_n = {}
    tests_n = {}
    for name in names:
        trains_n[name] = _reverse_dims(trains[name]).reshape(num_images,f_num)[:,:a_num]
        tests_n[name] = _reverse_dims(tests[name]).reshape(num_images,f_num)[:,:a_num]
    return trains_n, tests_n, a_num


# ---------------------------------------------------------------- cell 10
def build_regressor_training_set(trains_n, tests_n, split_index):
    n_ex = split_index
    train = torch.cat((trains_n["teacher"][:n_ex], tests_n["teacher"][:n_ex]), dim = 0)
    y = torch.cat((torch.zeros(n_ex), torch.ones(n_ex)), dim = 0)

    rand=torch.randperm(y.shape[0])
    train = train[rand]
    y = y[rand]
    return train, y


# ---------------------------------------------------------------- cell 11
def build_regressor(a_num):
    model = nn.Sequential(nn.Linear(a_num,100),nn.ReLU(),nn.Linear(100,1),nn.Tanh())
    criterion = nn.CrossEntropyLoss()
    optimizer =torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


# ---------------------------------------------------------------- cell 12
def train_regressor(model, optimizer, train, y, epochs=1000):
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            inputs = train
            outputs = model(inputs)
            loss = -1 * ((2*y-1)*(outputs.squeeze(-1))).mean()
            loss.backward()
            optimizer.step()
            pbar.set_description('loss {}'.format(loss.item()))
    return model


# ---------------------------------------------------------------- cell 14
def get_p(outputs_train, outputs_test):
    pred_test = outputs_test[:,0].detach().cpu().numpy()
    pred_train = outputs_train[:,0].detach().cpu().numpy()
    tval, pval = ttest_ind(pred_test, pred_train, alternative="greater", equal_var=False)
    if pval < 0:
        raise Exception(f"p-value={pval}")
    return pval

def get_p_values(num_ex, train, test, k):
    total = train.shape[0]
    sum_p = 0
    p_values = []
    positions_list = []
    for i in range(k):
        positions = torch.randperm(total)[:num_ex]
        p_val = get_p(train[positions], test[positions])
        positions_list.append(positions)
        p_values.append(p_val)
    return p_values

def get_fischer(num_ex, train, test, k):
    p_values = get_p_values(num_ex, train, test, k)
    return combine_pvalues(p_values, method="mudholkar_george")[1]

def get_max_p_value(num_ex, train, test, k):
    p_values = get_p_values(num_ex, train, test, k)
    return max(p_values)


# ---------------------------------------------------------------- cell 15
def score_features(model, trains_n, tests_n, names):
    outputs_tr = {}
    outputs_te = {}
    for name in names:
        outputs_tr[name] = model(trains_n[name])
        outputs_te[name] = model(tests_n[name])
    return outputs_tr, outputs_te


# ---------------------------------------------------------------- cell 16
def print_inference(outputs_train, outputs_test):
    m1, m2 = outputs_test[:,0].mean(), outputs_train[:,0].mean()
    pval = get_p(outputs_train, outputs_test)
    print(f"p-value = {pval} \t| Mean difference = {m1-m2}")

def inference_stats(outputs_train, outputs_test):
    """ADDED (g): the values print_inference prints, returned as floats."""
    m1, m2 = outputs_test[:,0].mean(), outputs_train[:,0].mean()
    pval = get_p(outputs_train, outputs_test)
    return float(pval), float((m1-m2).detach())


# ---------------------------------------------------------------- cell 17
def hold_out(outputs_tr, outputs_te, names, split_index):
    for name in names:
        outputs_tr[name], outputs_te[name] = outputs_tr[name][split_index:], outputs_te[name][split_index:]
    return outputs_tr, outputs_te


# ---------------------------------------------------------------- cell 20
def main_loop(outputs_tr, outputs_te, names, total_reps = 40, max_m = 45, total_inner_rep = 100):
    m_list = [x for x in range(2, max_m, 1)]
    p_values_all_threat_models_dict = {}

    n_pbar = tqdm(names, leave=False)
    for name in n_pbar:
        p_vals_per_rep_no = {}
        r_pbar = tqdm(range(total_reps), leave=False)
        for rep_no in r_pbar:
            p_values_list = []
            for m in m_list:
                p_list = get_p_values(m, outputs_tr[name], outputs_te[name], total_inner_rep)
                try:
                    hm = hmean(p_list)
                except:
                    hm = 1.0
                p_values_list.append(hm)
            r_pbar.set_description(f"rep_no: {rep_no+1}/{total_reps}")
            p_vals_per_rep_no[rep_no] = p_values_list
        p_values_all_threat_models_dict[name] = p_vals_per_rep_no

    df_list = []
    for name, rep_dict in p_values_all_threat_models_dict.items():
        df = pd.DataFrame(rep_dict).reset_index().assign(m=lambda r: r.index+2).drop(["index"], axis=1)
        df_list.append(pd.melt(df,id_vars=["m"], var_name="rep_no", value_name="p_value").assign(threat_model=name))
    results_df = pd.concat(df_list)
    return results_df


# ---------------------------------------------------- notebooks/utils.py
def generate_table(outputs_tr, outputs_te, names, selected_m=10, max_m=45, total_inner_rep=100, order=None):
    def get_p_mean_diff(outputs_train, outputs_test):
        pred_test = outputs_test[:,0].detach().cpu().numpy()
        pred_train = outputs_train[:,0].detach().cpu().numpy()
        tval, pval = ttest_ind(pred_test, pred_train, alternative="greater", equal_var=False)
        if pval < 0:
            raise Exception(f"p-value={pval}")  # original: ipdb.set_trace()
        return pval, (pred_test.mean() - pred_train.mean())

    def get_p_values_mean_diffs(num_ex, train, test, k):
        total = train.shape[0]
        sum_p = 0
        p_values = []
        diffs = []
        for i in range(k):
            positions = torch.randperm(total)[:num_ex]
            p_val,  mean_diff = get_p_mean_diff(train[positions], test[positions])
            p_values.append(p_val)
            diffs.append(mean_diff)
        return p_values, diffs

    name2p_val_mean_diff = {}

    n_pbar = tqdm(names, leave=False)  # original: tqdm.notebook.tqdm
    for name in n_pbar:
        p_values_list = []
        p_list, diffs = get_p_values_mean_diffs(selected_m, outputs_tr[name], outputs_te[name], total_inner_rep)
        try:
            hm = hmean(p_list)
        except:
            hm = 1.0
        diff = np.mean(diffs)
        name2p_val_mean_diff[name] = [hm, diff]

    tab = pd.DataFrame(name2p_val_mean_diff, index=["p_value", "mean_diff"]).T
    tab = tab[["mean_diff", "p_value"]]
    if order is None:
        order = ['teacher',
         'distillation',
         'pre-act-18',
         'zero-shot',
         'fine-tune',
         'extract-label',
         'extract-logit',
         'independent']

    tab = tab.loc[order]
    return tab
