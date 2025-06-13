import argparse
import scanpy as sc
import pandas as pd
import scipy.sparse

import json
import os
import numpy as np
import signal
import sys
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
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


def get_donor_dist(adata_path, genes, cell_types, synth_dists, donor, ct_dvgs=None):
    print(f"Donor {donor}")
    adata = sc.read_h5ad(adata_path)
    donor_mask = adata.obs.individual == donor
    cum_prob = 0
    n = 0
    for i, cell_row in enumerate(adata[donor_mask]):
        #print(f"{i} / {len(adata[donor_mask])}")
        cell_type = cell_row.obs.cell_type.iloc[0]
        for gene in genes:
            gene_val = cell_row[:,gene].X.toarray().flatten()[0]
            if gene in synth_dists[cell_type]:
                cum_prob += synth_dists[cell_type][gene].pmf(gene_val)
                n += 1

    print(cum_prob / n)
    if n == 0:
        return 0
    return cum_prob / n

def get_donor_dist_fast(adata, genes, synth_dists, donor):
    donor_mask = adata.obs["individual"] == donor
    donor_data = adata[donor_mask]

    gene_matrix = donor_data[:, genes].X

    if scipy.sparse.issparse(gene_matrix):
        gene_matrix = gene_matrix.toarray()

    # Get cell types for all cells
    cell_types = donor_data.obs["cell_type"].values

    total_prob = 0
    total_n = 0

    for i, gene in enumerate(genes):
        gene_vals = gene_matrix[:, i]
        for ct in synth_dists:
            if gene in synth_dists[ct]:
                ct_mask = (cell_types == ct)
                if np.any(ct_mask):
                    probs = synth_dists[ct][gene].pmf_vec(gene_vals[ct_mask])
                    total_prob += np.sum(probs)
                    total_n += np.sum(ct_mask)

    if total_n == 0:
        return 0

    return total_prob / total_n


def main(h5ad, hvg_file, splits_dir, ct_dvg_dir=None):
    adata = sc.read_h5ad(h5ad)
    donors = adata.obs.individual.unique()
    cell_types = adata.obs.cell_type.unique()
    hvg_raw = pd.read_csv(hvg_file)
    hvg_mask = hvg_raw["highly_variable"]
    genes = adata.var.index[hvg_mask]

    synth_dists = {}
    for i in range(1, 4):
        print(f"Calculating split {i}")
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

        donor_dist_partial = partial(get_donor_dist_fast, h5ad, genes, synth_dists)

        # You could speed this up a fair bit with multiprocessing, but that involves
        # dealing with multiple FPs to the H5AD, which is a pain. Vectorizing the
        # PMF function and iterating by genes led to a much bigger improvement
        # for much less effort.
        results = []
        for donor in tqdm(donors):
            prob = get_donor_dist_fast(adata, genes, synth_dists, donor)
            results.append(prob)

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
        print(f"\n\nSplit {i} results")
        print(f"AUROC: {auroc:.4f}")
        #print("Threshold\tTP\tFP")
        #for thr, tp, fp in zip(thresholds, TP, FP):
        #    print(f"{thr:.2f}\t\t{tp}\t{fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conduct MIA on all splits.")

    parser.add_argument("--genes", required=True, help="CSV with boolean 'highly_variable' column, genes index")
    parser.add_argument("--h5ad", required=True, help="Anndata containing expression for all donors")
    parser.add_argument("--splits", required=True, help="Dir with files synth_{i}.h5ad and donors_{i}.csv for each split of data")
    parser.add_argument("--dvg_dir", required=False, help="Dir containing donor variable genes used as output for kruskal.py")
    args = parser.parse_args()

    main(args.h5ad, args.genes, args.splits, args.dvg_dir)
