# experiments_resnet50.py
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from scipy.stats import median_abs_deviation as mad
from conformal import ConformalModelLogits
from utils import get_logits_dataset, get_model, split2, validate
from plotutils import plot_exp1


def exp1():
    """
    Runs your experiment with ResNet50 over multiple alpha values,
    compares Fixed, Naive, APS, RAPS. Returns a DataFrame 'df'.
    """

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    modelname = 'ResNet50'
    alphas = [0.05, 0.10, 0.20]  
    predictors = ['Fixed','Naive','APS','RAPS']

    datasetpath = "./datasets/imagenet-val"
    datasetname = "ImageNet"

    num_trials = 5

    kreg = None
    lamda = None
    randomized = True
    n_data_conf = 30000
    n_data_val = 20000
    pct_paramtune = 0.33
    bsz = 32

    results = []

    for alpha in alphas:
        for predictor in predictors:
            print(f"Running {modelname} at alpha={alpha}, predictor={predictor}")
            out = experiment(
                modelname, datasetname, datasetpath,
                num_trials, alpha, kreg, lamda, randomized,
                n_data_conf, n_data_val, pct_paramtune, bsz,
                predictor
            )
            top1, top5, coverage, size, _, _, _, _ = out
            results.append({
                "Model": modelname,
                "Predictor": predictor,
                "Alpha": alpha,  # store alpha for each run
                "Top1": round(top1,3),
                "Top5": round(top5,3),
                "Coverage": round(coverage,3),
                "Size": round(size,3),
            })

    df = pd.DataFrame(results)
    print("\nFinal Results:\n", df)

    df.rename(columns={"Alpha": "alpha"}, inplace=True)
    return df



def experiment(modelname, datasetname, datasetpath, num_trials, alpha,
               kreg, lamda, randomized, n_data_conf, n_data_val,
               pct_paramtune, bsz, predictor):
    naive_bool = (predictor == 'Naive')
    fixed_bool = (predictor == 'Fixed')
    if predictor in ['Fixed', 'Naive', 'APS']:
        kreg = 1
        lamda = 0

    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    model = get_model(modelname)

    top1s = np.zeros(num_trials)
    top5s = np.zeros(num_trials)
    coverages = np.zeros(num_trials)
    sizes = np.zeros(num_trials)

    for i in range(num_trials):
        top1_avg, top5_avg, cvg_avg, sz_avg = trial(
            model, logits, alpha, kreg, lamda, randomized,
            n_data_conf, n_data_val, pct_paramtune, bsz,
            naive_bool, fixed_bool
        )
        top1s[i] = top1_avg
        top5s[i] = top5_avg
        coverages[i] = cvg_avg
        sizes[i] = sz_avg

        print(f"[Trial {i+1}/{num_trials}] "
              f"Top1={np.median(top1s[:i+1]):.3f}, "
              f"Top5={np.median(top5s[:i+1]):.3f}, "
              f"Coverage={np.median(coverages[:i+1]):.3f}, "
              f"Size={np.median(sizes[:i+1]):.3f}")

    return (
        np.median(top1s),
        np.median(top5s),
        np.median(coverages),
        np.median(sizes),
        mad(top1s),
        mad(top5s),
        mad(coverages),
        mad(sizes),
    )



def trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf,
          n_data_val, pct_paramtune, bsz, naive_bool, fixed_bool):
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)

    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size=bsz, shuffle=False)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size=bsz, shuffle=False)

    if fixed_bool:
        import numpy as np
        gt_locs_cal = []
        for x_i in logits_cal:
            logits_np = x_i[0].cpu().numpy() if hasattr(x_i[0], 'cpu') else x_i[0]
            label = x_i[1].item() if hasattr(x_i[1], 'item') else x_i[1]

            sorted_indices = np.argsort(logits_np)
            sorted_indices = np.flip(sorted_indices)
            pos = np.where(sorted_indices == label)[0][0]
            gt_locs_cal.append(pos)
        gt_locs_cal = np.array(gt_locs_cal)

        gt_locs_val = []
        for x_i in logits_val:
            logits_np = x_i[0].cpu().numpy() if hasattr(x_i[0], 'cpu') else x_i[0]
            label = x_i[1].item() if hasattr(x_i[1], 'item') else x_i[1]

            sorted_indices = np.argsort(logits_np)
            sorted_indices = np.flip(sorted_indices)
            pos = np.where(sorted_indices == label)[0][0]
            gt_locs_val.append(pos)
        gt_locs_val = np.array(gt_locs_val)

        kstar = np.quantile(gt_locs_cal, 1 - alpha, interpolation='higher') + 1

        frac_numer = (gt_locs_cal <= (kstar-1)).mean() - (1 - alpha)
        frac_denom = ((gt_locs_cal <= (kstar-1)).mean()
                      - (gt_locs_cal <= (kstar-2)).mean())
        if frac_denom == 0:
            rand_frac = 0
        else:
            rand_frac = frac_numer / frac_denom

        sizes = np.ones_like(gt_locs_val) * (kstar - 1)
        sizes = sizes + (torch.rand(gt_locs_val.shape) > rand_frac).int().numpy()

        top1_avg = (gt_locs_val == 0).mean()
        top5_avg = (gt_locs_val <= 4).mean()
        cvg_avg = (gt_locs_val <= (sizes - 1)).mean()
        sz_avg = sizes.mean()

    else:
        conformal_model = ConformalModelLogits(
            model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda,
            randomized=randomized, allow_zero_sets=True,
            pct_paramtune=pct_paramtune, naive=naive_bool,
            batch_size=bsz, lamda_criterion='size'
        )
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(loader_val, conformal_model, print_bool=False)

    return top1_avg, top5_avg, cvg_avg, sz_avg



if __name__ == "__main__":
    import os

    os.makedirs("./.cache", exist_ok=True)
    cache_fname = "./.cache/resnet50_experiment.csv"

    if os.path.exists(cache_fname):
        df = pd.read_csv(cache_fname)
        print("Loaded cached results from", cache_fname)
    else:
        df = exp1()
        df.to_csv(cache_fname, index=False)
        print("Experiment finished and results cached to", cache_fname)

    from plotutils import plot_exp1
    plot_exp1(df)
