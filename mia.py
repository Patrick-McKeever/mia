import argparse
import scanpy as sc
import pandas as pd

import json
import os
import numpy as np
import signal
import sys
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import roc_auc_score, roc_curve
from distribution import (CountDistribution, load_distribution_from_json,
                          save_distribution_to_json, load_distribution_from_dict)


def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def kl_divergence(P: CountDistribution, Q: CountDistribution, k_max=100):
    kl = 0.0
    for k in range(k_max + 1):
        p_k = P.pmf(k)
        q_k = Q.pmf(k)

        if p_k > 0:
            if q_k == 0:
                # KL divergence is infinite if Q(k)=0 and P(k)>0
                return float('inf')
            kl += p_k * np.log(p_k / q_k)
    return kl


def assign_rank_probabilities(d):
    if not d:
        return {}

    n = len(d) - 1
    if n == 0:
        return {k: 1.0 for k in d}

    sorted_items = sorted(d.items(), key=lambda item: item[1])
    result = {}
    for i, (k, _) in enumerate(sorted_items):
        result[k] = i / n
    return result


def get_donor_dist(genes, cell_types, synth_dists, donor, ct_dvgs=None):
    print(f"Donor {donor}")
    score = 0
    non_inf = 0
    path = f"donor_dists/{donor}.json"
    with open(path, "r") as f:
        donor_dists = json.load(f)

    for cell_type_int in cell_types:
        # JSON encoding has int keys as strs.
        cell_type = str(cell_type_int)
        for gene in genes:
            if ct_dvgs is not None:
                if not ct_dvgs[cell_type_int].loc[gene]["significant"]:
                    continue
            if cell_type not in synth_dists:
                continue
            if gene not in synth_dists[cell_type]:
                continue
            if str(cell_type) not in donor_dists:
                continue
            if gene not in donor_dists[str(cell_type)]:
                continue
            if os.path.exists(path):
                donor_dist = load_distribution_from_dict(donor_dists[str(cell_type)][gene])
                synth_dist = synth_dists[cell_type][gene]
                div = kl_divergence(synth_dist, donor_dist)
                if div < float('inf'):
                    score += div
                    non_inf += 1

    if non_inf == 0:
        return 0

    with open(f"donor_scores/{donor}", "w+") as f:
        f.write(str(score / non_inf))
    return score / non_inf


def main(h5ad, hvg_file, splits_dir, ct_dvg_dir=None):
    adata = sc.read_h5ad(h5ad)
    donors = adata.obs.individual.unique()
    cell_types = adata.obs.cell_type.unique()
    hvg_raw = pd.read_csv(hvg_file)
    hvg_mask = hvg_raw["highly_variable"]
    genes = adata.var.index[hvg_mask]

    synth_dists = {}
    for i in range(1, 4):
        donors_df = pd.read_csv(f"{splits_dir}/donors_{i}.csv")
        in_synth = set(donors_df.iloc[:, 1])

        donor_probs_true = {}
        for ind in donors:
            donor_probs_true[ind] = 0
            if ind in in_synth:
                donor_probs_true[ind] = 1

        ct_dvgs = None
        if ct_dvg_dir is not None:
            ct_dvgs = {}
            for ct in cell_types:
                ct_dvg_path = os.path.join(ct_dvg_dir, f"{ct}.csv")
                ct_dvgs[ct] = pd.read_csv(ct_dvg_path)
                ct_dvgs[ct].set_index("gene", inplace=True)

        for cell_type in cell_types:
            synth_dists[cell_type] = {}
            for j, gene in enumerate(genes):
                path = os.path.join("synth_dist", f"synth_{i}_{gene}_{cell_type}.json")
                if not os.path.exists(path):
                    continue
                try:
                    synth_dists[cell_type][gene] = load_distribution_from_json(path)
                except RuntimeError:
                    pass

        donor_dist_partial = partial(get_donor_dist, genes, cell_types, synth_dists)
        with Pool(processes=20) as executor:
            results = executor.map(donor_dist_partial, donors)

        donor_scores = dict(zip(donors, results))
        with open("donor_scores.json", "w+") as f:
            json.dump(donor_scores, f)

        predicted_probs = list(assign_rank_probabilities(donor_scores).values())

        y_true = list(donor_probs_true.values())

        auroc = roc_auc_score(y_true, predicted_probs)
        fpr, tpr, thresholds = roc_curve(y_true, predicted_probs)

        TP = (tpr * sum(y_true)).astype(int)
        FP = (fpr * sum(1 - np.array(y_true))).astype(int)

        # Print results
        print(f"\n\nSplit {i}")
        print(f"AUROC: {auroc:.4f}")
        print("Threshold\tTP\tFP")
        for thr, tp, fp in zip(thresholds, TP, FP):
            print(f"{thr:.2f}\t\t{tp}\t{fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conduct MIA on all splits.")

    parser.add_argument("--genes", required=True, help="CSV with boolean 'highly_variable' column, genes index")
    parser.add_argument("--h5ad", required=True, help="Anndata containing expression for all donors")
    parser.add_argument("--splits", required=True, help="Dir with files synth_{i}.h5ad and donors_{i}.csv for each split of data")
    parser.add_argument("--dvg_dir", required=False, help="Dir containing donor variable genes used as output for kruskal.py")
    args = parser.parse_args()

    main(args.h5ad, args.genes, args.splits, args.dvg_dir)
