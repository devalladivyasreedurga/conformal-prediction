# conformal.py
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import time
from tqdm import tqdm
from utils import validate, get_logits_targets, sort_sum, device
import pdb



class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None,
                 randomized=True, allow_zero_sets=False, pct_paramtune=0.3,
                 batch_size=32, lamda_criterion='size'):
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.T = torch.tensor([1.3], device=device)  # initialize (1.3 is usually a good value)
        self.T, calib_logits = platt(self, calib_loader)
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = len(calib_loader.dataset.dataset.classes)

        if kreg is None or lamda is None:
            kreg, lamda, calib_logits = pick_parameters(
                model, calib_logits, alpha, kreg, lamda, randomized,
                allow_zero_sets, pct_paramtune, batch_size, lamda_criterion
            )

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda

        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size,
                                        shuffle=False, pin_memory=True)

        self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, x, randomized=None, allow_zero_sets=None):
        if randomized is None:
            randomized = self.randomized
        if allow_zero_sets is None:
            allow_zero_sets = self.allow_zero_sets

        # Move input to device
        x = x.to(device)
        logits = self.model(x)
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy / self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)
            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum,
                    penalties=self.penalties, randomized=randomized,
                    allow_zero_sets=allow_zero_sets)

        return logits, S


def conformal_calibration(cmodel, calib_loader):
    print("Conformal calibration")
    E = np.array([])
    with torch.no_grad():
        for x, targets in tqdm(calib_loader):
            x = x.to(device)
            logits = cmodel.model(x).cpu().numpy()
            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)
            E = np.concatenate((
                E,
                giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum,
                    penalties=cmodel.penalties, randomized=True,
                    allow_zero_sets=True)
            ))

    Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')
    return Qhat

def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    print("Begin Platt scaling.")
    logits_dataset = get_logits_targets(cmodel.model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(logits_dataset,
                                                batch_size=calib_loader.batch_size,
                                                shuffle=False, pin_memory=True)

    T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)
    print(f"Optimal T={T.item()}")
    return T, logits_dataset


class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None,
                 randomized=True, allow_zero_sets=False, naive=False, LAC=False,
                 pct_paramtune=0.3, batch_size=32, lamda_criterion='size'):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        self.T = platt_logits(self, calib_loader)

        if (kreg is None or lamda is None) and not naive and not LAC:
            kreg, lamda, calib_logits = pick_parameters(
                model, calib_loader.dataset, alpha, kreg, lamda,
                randomized, allow_zero_sets, pct_paramtune,
                batch_size, lamda_criterion
            )
            calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size,
                                            shuffle=False, pin_memory=True)

        # E.g. if each sample is (logits_vector, label), we assume 1000 classes:
        num_classes = calib_loader.dataset[0][0].shape[0]
        self.penalties = np.zeros((1, num_classes))

        if kreg is not None and not naive and not LAC:
            self.penalties[:, kreg:] += lamda

        self.Qhat = 1 - alpha
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader)
        elif not naive and LAC:
            gt_locs_cal = np.array([
                np.where(np.argsort(x[0])[::-1] == x[1])[0][0]
                for x in calib_loader.dataset
            ])
            # Example with .flip replaced by [::-1]
            scores_cal = 1 - np.array([
                np.sort(torch.softmax(calib_loader.dataset[i][0]/self.T.item(), dim=0))[::-1][gt_locs_cal[i]]
                for i in range(len(calib_loader.dataset))
            ])
            self.Qhat = np.quantile(
                scores_cal,
                np.ceil((scores_cal.shape[0] + 1) * (1 - alpha)) / scores_cal.shape[0]
            )

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized is None:
            randomized = self.randomized
        if allow_zero_sets is None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy / self.T.item(), axis=1)

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)
                S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum,
                        penalties=self.penalties, randomized=randomized,
                        allow_zero_sets=allow_zero_sets)
            else:
                S = [
                    np.where((1 - scores[i, :]) < self.Qhat)[0]
                    for i in range(scores.shape[0])
                ]

        return logits, S

def conformal_calibration_logits(cmodel, calib_loader):
    E = np.array([])
    with torch.no_grad():
        for logits, targets in calib_loader:
            logits = logits.cpu().numpy()
            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)
            E = np.concatenate((
                E,
                giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum,
                    penalties=cmodel.penalties, randomized=True,
                    allow_zero_sets=True)
            ))

    Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')
    return Qhat

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    """
    Fits a temperature parameter T by cross-entropy on the calibration set.
    """
    # Put T on the correct device
    T = nn.Parameter(torch.tensor([1.3], device=device))

    nll_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD([T], lr=lr)

    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            x = x.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # "out" is x / T, meaning each sample is already logits => we divide by T
            out = x / T
            loss = nll_criterion(out, targets.long())
            loss.backward()
            optimizer.step()

        if abs(T_old - T.item()) < epsilon:
            break

    return T


def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1
    sizes_base = np.minimum(sizes_base, scores.shape[1])  # clamp to 1000 classes

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            idx = sizes_base[i] - 1
            # Avoid negative indexing if sizes_base[i] is 0
            if idx < 0:
                idx = 0
            # same logic as original code
            V[i] = (1 / ordered[i, idx]) * (
                tau - (cumsum[i, idx] - ordered[i, idx]) - penalties_cumsum[0, idx]
            )

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    # If tau == 1.0, always predict max size
    if tau == 1.0:
        sizes[:] = cumsum.shape[1]

    # Disallow zero sets if user chooses
    if not allow_zero_sets:
        sizes[sizes == 0] = 1

    S = []
    for i in range(I.shape[0]):
        S.append(I[i, 0:sizes[i]])
    return S

def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets):
    idx = np.where(I == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random()
    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return (U * ordered[idx]
                + cumsum[(idx[0], idx[1]-1)]
                + penalty[0:(idx[1][0]+1)].sum())

def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    E = -np.ones(scores.shape[0])
    for i in range(scores.shape[0]):
        E[i] = get_tau(
            scores[i:i+1, :],
            targets[i].item(),
            I[i:i+1, :],
            ordered[i:i+1, :],
            cumsum[i:i+1, :],
            penalties[0, :],
            randomized=randomized,
            allow_zero_sets=allow_zero_sets
        )
    return E



def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = []
    for x_i in paramtune_logits:
        if hasattr(x_i[0], "cpu"):
            logits_np = x_i[0].cpu().numpy()
        else:
            logits_np = x_i[0]

        if hasattr(x_i[1], "item"):
            label = x_i[1].item()
        else:
            label = x_i[1]

        sorted_indices = np.argsort(logits_np)
        sorted_indices = np.flip(sorted_indices)  

        pos = np.where(sorted_indices == label)[0][0]
        gt_locs_kstar.append(pos)

    gt_locs_kstar = np.array(gt_locs_kstar)

    kstar = np.quantile(gt_locs_kstar, 1 - alpha, method='higher') + 1

    return kstar


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    
    first_batch = next(iter(paramtune_loader))
    # first_batch[0] => logits, shape = [batch_size, 1000]
    num_classes = first_batch[0].shape[1] if hasattr(first_batch[0], 'shape') else 1000
    best_size = num_classes

    lamda_star = 0.0
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:
        conformal_model = ConformalModelLogits(
            model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
            randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False
        )
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star

def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets,
                            strata=[[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]):
    lamda_star = 0
    best_violation = 1
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]:
        conformal_model = ConformalModelLogits(
            model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
            randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False
        )
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star

def pick_parameters(model, calib_logits, alpha, kreg, lamda,
                    randomized, allow_zero_sets, pct_paramtune,
                    batch_size, lamda_criterion):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(
        calib_logits,
        [num_paramtune, len(calib_logits) - num_paramtune]
    )
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size,
                                    shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size,
                                        shuffle=False, pin_memory=True)

    if kreg is None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda is None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg,
                                    randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg,
                                            randomized, allow_zero_sets)
    return kreg, lamda, calib_logits

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit)
        size_arr = np.array([x.size for x in S])
        _, _, _ = sort_sum(logit.numpy())
        correct = np.zeros_like(size_arr)
        for j in range(correct.shape[0]):
            if target[j] in S[j]:
                correct[j] = 1
        batch_df = pd.DataFrame({'size': size_arr, 'correct': correct})
        df = pd.concat([df, batch_df], ignore_index=True)

    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df['size'] >= stratum[0]) & (df['size'] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation
