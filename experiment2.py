import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import itertools

from conformal import ConformalModelLogits
from utils import get_logits_dataset, get_model, validate


def trial(model, logits, alpha, kreg, lamda, randomized,
          n_data_conf, n_data_val, bsz, naive_bool, fixed_bool):
    """
    Splits 'logits' into:
      - calibration subset of size n_data_conf
      - validation subset of size n_data_val
    Then measures coverage & average set size, ignoring accuracy.
    """

    import torch.utils.data as tdata

    total_len = len(logits)
    if n_data_conf + n_data_val > total_len:
        raise ValueError(f"Not enough data. n_data_conf + n_data_val = {n_data_conf + n_data_val} > {total_len}")

    # 1. Randomly split for calibration
    cal_size = n_data_conf
    remainder_size = total_len - cal_size
    subset_cal, subset_remainder = tdata.random_split(
        logits,
        [cal_size, remainder_size]
    )

    # 2. Randomly split remainder for validation
    val_size = n_data_val
    leftover_size = remainder_size - val_size
    subset_val, _ = tdata.random_split(
        subset_remainder,
        [val_size, leftover_size]
    )

    loader_cal = tdata.DataLoader(subset_cal, batch_size=bsz, shuffle=False)
    loader_val = tdata.DataLoader(subset_val, batch_size=bsz, shuffle=False)

   
    if fixed_bool:
        gt_locs_cal = []
        for x_i in subset_cal:
            logits_np = x_i[0].cpu().numpy() if hasattr(x_i[0], 'cpu') else x_i[0]
            label = x_i[1].item() if hasattr(x_i[1], 'item') else x_i[1]

            sorted_indices = np.argsort(logits_np)      # ascending
            sorted_indices = np.flip(sorted_indices)    # descending
            pos = np.where(sorted_indices == label)[0][0]
            gt_locs_cal.append(pos)
        gt_locs_cal = np.array(gt_locs_cal)

        gt_locs_val = []
        for x_i in subset_val:
            logits_np = x_i[0].cpu().numpy() if hasattr(x_i[0], 'cpu') else x_i[0]
            label = x_i[1].item() if hasattr(x_i[1], 'item') else x_i[1]

            sorted_indices = np.argsort(logits_np)
            sorted_indices = np.flip(sorted_indices)
            pos = np.where(sorted_indices == label)[0][0]
            gt_locs_val.append(pos)
        gt_locs_val = np.array(gt_locs_val)

        kstar = np.quantile(gt_locs_cal, 1 - alpha, interpolation='higher') + 1

        frac_numer = (gt_locs_cal <= (kstar - 1)).mean() - (1 - alpha)
        frac_denom = (gt_locs_cal <= (kstar - 1)).mean() - (gt_locs_cal <= (kstar - 2)).mean()
        if frac_denom == 0:
            rand_frac = 0
        else:
            rand_frac = frac_numer / frac_denom

        import torch
        sizes = np.ones_like(gt_locs_val) * (kstar - 1)
        sizes = sizes + (torch.rand(gt_locs_val.shape) > rand_frac).int().numpy()

        coverage = (gt_locs_val <= (sizes - 1)).mean()
        size = sizes.mean()
        return coverage, size

   
    else:
        conformal_model = ConformalModelLogits(
            model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda,
            randomized=randomized, allow_zero_sets=True,
            naive=naive_bool, batch_size=bsz, lamda_criterion='size'
        )
        # "validate" returns (top1, top5, coverage, size) but we only want coverage & size
        top1_, top5_, coverage, size = validate(loader_val, conformal_model, print_bool=False)
        return coverage, size


def experiment(modelname, datasetname, datasetpath,
               num_trials, alpha, kreg, lamda, randomized,
               n_data_conf, n_data_val, bsz, predictor):
    """
    Repeats 'trial' multiple times, returning the median coverage & size.
    Ignores accuracy entirely.
    """
    naive_bool = (predictor == 'Naive')
    fixed_bool = (predictor == 'Fixed')
    if predictor in ['Fixed','Naive','APS']:
        kreg = 1
        lamda = 0

    # 1. Load entire dataset's logits
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    # 2. Build model for conformal if needed
    model = get_model(modelname)

    import numpy as np
    coverages = np.zeros(num_trials)
    sizes = np.zeros(num_trials)

    for i in range(num_trials):
        cvg, sz = trial(
            model, logits, alpha, kreg, lamda, randomized,
            n_data_conf, n_data_val, bsz,
            naive_bool, fixed_bool
        )
        coverages[i] = cvg
        sizes[i] = sz
        print(f"Trial {i+1}/{num_trials} => coverage={np.median(coverages[:i+1]):.3f}, size={np.median(sizes[:i+1]):.3f}")

    from scipy.stats import median_abs_deviation as mad
    coverage_med = np.median(coverages)
    size_med = np.median(sizes)
    coverage_mad = mad(coverages)
    size_mad = mad(sizes)
    return coverage_med, size_med, coverage_mad, size_mad


if __name__ == "__main__":
    import os
    os.makedirs("./outputs", exist_ok=True)

    # We'll do multiple alpha
    alphas = [0.05, 0.1, 0.2]

    # We'll compare these methods
    predictors = ['Fixed','Naive','APS','RAPS']

    # We'll do ResNet50
    modelname = 'ResNet50'

    # Trials
    num_trials = 2

    # Conformal/logit parameters
    n_data_conf = 5000
    n_data_val = 5000
    bsz = 32
    kreg = None
    lamda = None
    randomized = True

    # Our dataset path
    datasetname = "ImageNetV2"
    datasetpath = "exp2-data/imagenetV2/imagenetv2"

    import pandas as pd
    results_df = pd.DataFrame(columns=["Model","Predictor","Alpha","Coverage","Size"])

    for alpha in alphas:
        for predictor in predictors:
            coverage_med, size_med, coverage_mad, size_mad = experiment(
                modelname=modelname,
                datasetname=datasetname,
                datasetpath=datasetpath,
                num_trials=num_trials,
                alpha=alpha,
                kreg=kreg,
                lamda=lamda,
                randomized=randomized,
                n_data_conf=n_data_conf,
                n_data_val=n_data_val,
                bsz=bsz,
                predictor=predictor
            )

            row = {
                "Model": modelname,
                "Predictor": predictor,
                "Alpha": alpha,
                "Coverage": round(coverage_med,3),
                "Size": round(size_med,3)
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    print("\nFinal coverage vs. set size (no accuracies):\n")
    print(results_df)
    results_df.to_csv("./outputs/experiment2_imagenetv2.csv", index=False)
    print("Saved to ./outputs/experiment2_imagenetv2.csv")
