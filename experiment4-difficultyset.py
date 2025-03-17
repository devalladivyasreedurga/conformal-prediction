import os, sys

# Ensure we can import local modules (utils.py, conformal.py)
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import itertools

from conformal import ConformalModelLogits
from utils import get_logits_dataset, get_model, split2, sort_sum

###############################################################################
# Plotting / Table function
###############################################################################
def difficulty_table(df_big):
    """
    Builds a LaTeX table showing coverage and average set size
    stratified by difficulty (i.e., the top-k rank of the true label).
    """
    topks = [[1,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]

    tbl = ""
    tbl += "\\begin{table}[t]\n"
    tbl += "\\centering\n"
    tbl += "\\tiny\n"
    tbl += "\\begin{tabular}{lc"

    lamdaunique = df_big.lamda.unique()
    lamdaunique.sort()

    multicol_line = "       & " 
    midrule_line = "        "
    label_line = "difficulty & count "

    # Build columns for each lambda
    for i in range(len(lamdaunique)):
        j = 2*i
        tbl += "cc"
        multicol_line += f" & \\multicolumn{{2}}{c}{{$\\lambda={lamdaunique[i]}$}}    "
        midrule_line += f" \\cmidrule(r){{{j+3}-{j+4}}}    "
        label_line += "& cvg & sz "

    tbl += "} \n"
    tbl += "\\toprule\n"
    multicol_line += "\\\\ \n"
    midrule_line += "\n"
    label_line += "\\\\ \n"

    tbl += multicol_line + midrule_line + label_line
    tbl += "\\midrule \n"

    # For each difficulty bin, compute coverage & average size
    for topk_range in topks:
        if topk_range[0] == topk_range[1]:
            tbl += f"{topk_range[0]}     "
        else:
            tbl += f"{topk_range[0]} to {topk_range[1]}     "

        # Subset main DataFrame for the relevant topk range
        df_bin = df_big[(df_big.topk >= topk_range[0]) & (df_big.topk <= topk_range[1])]

        # The count is for 1 lamda, so we divide by len(lamdaunique)
        total_count = int(len(df_bin) / len(lamdaunique))
        tbl += f" & {total_count} "

        for lamda in lamdaunique:
            df_small = df_bin[df_bin.lamda == lamda]
            if len(df_small) == 0:
                tbl += " & 0.00 & 0.0 "
                continue

            # coverage = fraction with topk <= size
            cvg = (df_small[df_small.topk <= df_small['size']].shape[0]) / len(df_small)
            sz = df_small['size'].mean()

            tbl += f" & {cvg:.2f} & {sz:.1f} "

        tbl += "\\\\ \n"

    tbl += "\\bottomrule\n"
    tbl += "\\end{tabular}\n"
    tbl += "\\caption{\\textbf{Coverage and size conditional on difficulty.} We report coverage and size of \\raps\\ sets for ResNet-152 with $k_{reg}=5$ and varying $\\lambda$.}\n"
    tbl += "\\label{table:difficulty}\n"
    tbl += "\\end{table}\n"

    return tbl

###############################################################################
# Main function to measure coverage & size for each difficulty bin
###############################################################################
def sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda,
               randomized, n_data_conf, n_data_val, bsz, predictor):
    """
    Returns a DataFrame with:
      - 'size': the set size for each test image
      - 'topk': the rank of the true label
      - 'lamda': the lambda used
    """
    _fix_randomness()

    naive_bool = (predictor == 'Naive')
    lamda_predictor = lamda
    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # no regularization for Naive/APS

    # Load or compute logits
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    # Split for calibration + validation
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)

    # Data loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size=bsz, shuffle=False, pin_memory=True)

    # Build model & conformal wrapper
    model = get_model(modelname)
    conformal_model = ConformalModelLogits(
        model, loader_cal, alpha=alpha, kreg=kreg,
        lamda=lamda_predictor, randomized=randomized,
        allow_zero_sets=True, naive=naive_bool
    )

    df = pd.DataFrame(columns=['model','predictor','size','topk','lamda'])
    corrects = 0
    denom = 0

    for (logit, target) in tqdm(loader_val, desc="Validating"):
        output, S = conformal_model(logit)
        size_arr = np.array([x.size for x in S])

        I, _, _ = sort_sum(logit.numpy())
        # rank of the true label
        topk_arr = np.where((I - target.view(-1,1).numpy()) == 0)[1] + 1

        corrects += sum(topk_arr <= size_arr)
        denom += output.shape[0]

        batch_df = pd.DataFrame({
            'model': modelname,
            'predictor': predictor,
            'size': size_arr,
            'topk': topk_arr,
            'lamda': lamda
        })
        # OLD: df = df.append(batch_df, ignore_index=True)
        df = pd.concat([df, batch_df], ignore_index=True)

    print(f"Empirical coverage: {corrects/denom:.3f} with lambda={lamda}")
    return df

def _fix_randomness(seed=0):
    """Fix relevant random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # If on Apple Silicon, you might remove or comment out `.cuda()` calls in get_model(...).

###############################################################################
# Main script logic
###############################################################################
if __name__ == "__main__":
    import os

    # Ensure outputs directory
    os.makedirs("./outputs", exist_ok=True)

    # Experiment config
    modelnames = ['ResNet152']
    alphas = [0.1]
    predictors = ['RAPS']
    lamdas = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 1]

    params = list(itertools.product(modelnames, alphas, predictors, lamdas))

    datasetname = 'Imagenet'
    # Adjust to your local path if needed:
    datasetpath = "./datasets/imagenet-val"

    kreg = 5
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    cudnn.benchmark = True

    # We'll store final DataFrame across all runs
    df = pd.DataFrame(columns=["model","predictor","size","topk","lamda"])

    for (modelname, alpha, predictor, lamda) in params:
        print(f"Model={modelname}, coverage={1-alpha}, predictor={predictor}, lambda={lamda}")
        out = sizes_topk(modelname, datasetname, datasetpath, alpha,
                         kreg, lamda, randomized, n_data_conf,
                         n_data_val, bsz, predictor)
        # OLD: df = df.append(out, ignore_index=True)
        df = pd.concat([df, out], ignore_index=True)

    # Build the difficulty table
    tbl = difficulty_table(df)
    print(df)
    print(tbl)

    # Save to disk
    with open("./outputs/difficulty_table.tex", "w") as f:
        f.write(tbl)

    print("Done. Table saved to ./outputs/difficulty_table.tex")
