import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from conformal import ConformalModelLogits
from utils import get_logits_dataset, get_model, split2, sort_sum

def adaptiveness_table(df_big):
    """
    Generates a LaTeX table summarizing coverage conditional on set size.
    Splits set sizes into bins, then calculates coverage for each bin.
    """
    sizes = [[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]

    tbl = ""
    tbl += "\\begin{table}[t]\n"
    tbl += "\\centering\n"
    tbl += "\\small\n"
    tbl += "\\begin{tabular}{l"

    lamdaunique = df_big.lamda.unique()
    lamdaunique.sort() 

    multicol_line = "        " 
    midrule_line = "        "
    label_line = "size "

    for i in range(len(lamdaunique)):
        j = 2*i 
        tbl += "cc"
        multicol_line += (" & \\multicolumn{2}{c}{$\\lambda={" + str(lamdaunique[i]) + "}$}    ")
        midrule_line += (" \\cmidrule(r){" + str(j+2) + "-" + str(j+3) + "}    ")
        label_line += "&cnt &cvg "

    tbl += "} \n"
    tbl += "\\toprule\n"
    multicol_line += "\\\\ \n"
    midrule_line += "\n"
    label_line += "\\\\ \n"

    tbl = tbl + multicol_line + midrule_line + label_line
    tbl += "\\midrule \n"

    total_coverages = {lamda: 0 for lamda in lamdaunique}

    for sz in sizes:
        if sz[0] == sz[1]:
            tbl += f"{sz[0]}     "
        else:
            tbl += f"{sz[0]} to {sz[1]}     "

        df_bin = df_big[(df_big['size'] >= sz[0]) & (df_big['size'] <= sz[1])]

        for lamda in lamdaunique:
            df_small = df_bin[df_bin.lamda == lamda]
            if len(df_small) == 0:
                tbl += " & 0 & "
                continue

            cvg = (df_small[df_small.topk <= df_small['size']].shape[0]) / len(df_small)
            tbl += f" & {len(df_small)} & {cvg:.2f}"

        tbl += "\\\\ \n"

    tbl += "\\bottomrule\n"
    tbl += "\\end{tabular}\n"
    tbl += "\\caption{\\textbf{Coverage conditional on set size.} We report average coverage of images stratified by the size of the set output by a conformalized ResNet-152 for $k_{reg}=5$ and varying $\\lambda$.}\n"
    tbl += "\\label{table:adaptiveness}\n"
    tbl += "\\end{table}\n"

    return tbl


def _fix_randomness(seed=0):
    """Fix all relevant random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
def sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda, randomized,
               n_data_conf, n_data_val, bsz, predictor):
    """
    Returns a DataFrame with one row per test image, containing:
      - 'size': the set size for that image
      - 'topk': the rank of the true label among predicted classes
      - 'lamda': the current lambda value
    """
    _fix_randomness()

    naive_bool = (predictor == 'Naive')
    lamda_predictor = lamda
    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0  # No regularization for Naive/APS

    # Load or compute logits
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    # Split for calibration + validation
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)

    # Prepare data loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size=bsz, shuffle=False, pin_memory=True)

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
        # forward pass
        output, S = conformal_model(logit)
        size = np.array([x.size for x in S])

        # rank of the true label
        I, _, _ = sort_sum(logit.numpy())
        topk = np.where((I - target.view(-1,1).numpy())==0)[1] + 1

        # track coverage
        corrects += sum(topk <= size)
        denom += output.shape[0]

        batch_df = pd.DataFrame({
            'model': modelname,
            'predictor': predictor,
            'size': size,
            'topk': topk,
            'lamda': lamda
        })
        # OLD: df = df.append(batch_df, ignore_index=True)
        df = pd.concat([df, batch_df], ignore_index=True)

    print(f"Empirical coverage: {corrects/denom:.3f} with lambda: {lamda}")
    return df



if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    modelnames = ['ResNet50']
    alphas = [0.1]
    predictors = ['RAPS']
    lamdas = [0, 0.001, 0.01, 0.1, 1]

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

    # We'll store the final DataFrame across all runs
    df = pd.DataFrame(columns=["model","predictor","size","topk","lamda"])

    # Loop over all parameter combos
    for (modelname, alpha, predictor, lamda) in params:
        print(f"Model={modelname}, coverage={1-alpha}, predictor={predictor}, lambda={lamda}")
        out = sizes_topk(modelname, datasetname, datasetpath,
                         alpha, kreg, lamda, randomized,
                         n_data_conf, n_data_val, bsz, predictor)
        # OLD: df = df.append(out, ignore_index=True)
        df = pd.concat([df, out], ignore_index=True)

    # Generate the adaptiveness table
    tbl = adaptiveness_table(df)
    print(tbl)
    print(df)
    # Write the table to disk
    with open("./outputs/adaptiveness_table.tex", 'w') as f:
        f.write(tbl)

    print("Done. The table is in ./outputs/adaptiveness_table.tex")
