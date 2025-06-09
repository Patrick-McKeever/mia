"""
Code for fitting distributions per gene, per cell type.
"""

import numpy as np
import argparse
import json
import os
import scanpy as sc
import pandas as pd
from scipy.stats import poisson, nbinom, chi2
from datetime import datetime
from scipy.optimize import minimize
from distribution import (CountDistribution, NegativeBinomial, ZeroInflatedNegativeBinomial,
                          Poisson, ZeroInflatedPoisson, save_distribution_to_json)


def negloglik_poisson(params, data):
    lam = params[0]
    if lam <= 0:
        return np.inf
    return -np.sum(poisson.logpmf(data, lam))


def negloglik_zip(params, data):
    pi, lam = params
    if not (0 < pi < 1) or lam <= 0:
        return np.inf
    loglik = np.where(
        data == 0,
        np.log(pi + (1 - pi) * poisson.pmf(0, lam)),
        np.log((1 - pi) * poisson.pmf(data, lam))
    )
    return -np.sum(loglik)


def negloglik_nb(params, data):
    r, p = params
    if r <= 0 or not (0 < p < 1):
        return np.inf
    return -np.sum(nbinom.logpmf(data, r, p))


def negloglik_zinb(params, data):
    pi, r, p = params
    if not (0 < pi < 1) or r <= 0 or not (0 < p < 1):
        return np.inf
    loglik = np.where(
        data == 0,
        np.log(pi + (1 - pi) * nbinom.pmf(0, r, p)),
        np.log((1 - pi) * nbinom.pmf(data, r, p))
    )
    return -np.sum(loglik)


def fit_counts_model(data, pval_cutoff=0.05):
    data = np.asarray(data)
    mu = np.mean(data)
    var = np.var(data, ddof=1)

    overdispersed = var > mu

    if not overdispersed:
        # Fit Poisson
        poisson_mle = [mu]
        ll_poisson = -negloglik_poisson(poisson_mle, data)

        # Fit ZIP
        res_zip = minimize(negloglik_zip, x0=[0.1, mu], args=(data,),
                           bounds=[(1e-5, 1-1e-5), (1e-5, None)], method='L-BFGS-B')
        if not res_zip.success:
            return Poisson(mu)
        ll_zip = -res_zip.fun

        # LRT: Poisson vs ZIP
        chisq = 2 * (ll_zip - ll_poisson)
        pval = 1 - chi2.cdf(chisq, df=1)

        if pval < pval_cutoff:
            return ZeroInflatedPoisson(res_zip.x[0], res_zip.x[1])
        else:
            return Poisson(mu)

    else:
        # Fit NB
        mu = np.mean(data)
        var = np.var(data, ddof=1)
        r_init = mu**2 / (var - mu) if var > mu else 10
        p_init = r_init / (r_init + mu)

        res_nb = minimize(negloglik_nb, x0=[r_init, p_init], args=(data,),
                          bounds=[(1e-5, None), (1e-5, 1 - 1e-5)], method='L-BFGS-B')
        if not res_nb.success:
            raise RuntimeError("NB fit failed")

        r_mle, p_mle = res_nb.x
        ll_nb = -res_nb.fun

        # Fit ZINB
        res_zinb = minimize(negloglik_zinb, x0=[0.1, r_mle, p_mle], args=(data,),
                            bounds=[(1e-5, 1 - 1e-5), (1e-5, None), (1e-5, 1 - 1e-5)],
                            method='L-BFGS-B')
        if not res_zinb.success:
            return NegativeBinomial(r_mle, p_mle)
        ll_zinb = -res_zinb.fun

        # LRT: NB vs ZINB
        chisq = 2 * (ll_zinb - ll_nb)
        pval = 1 - chi2.cdf(chisq, df=1)

        if pval < pval_cutoff:
            return ZeroInflatedNegativeBinomial(res_zinb.x[0], res_zinb.x[1], res_zinb.x[2])
        else:
            return NegativeBinomial(r_mle, p_mle)


def fit_and_save_distribution(gene, cell_type, train_adata, out_path):
    if not os.path.exists(out_path):
        cell_type_mask = train_adata.obs.cell_type == cell_type
        counts = train_adata[cell_type_mask, gene].X.toarray().flatten()
        if len(counts) == 0 or counts.mean() == 0:
            return
        dist = fit_counts_model(counts)
        save_distribution_to_json(dist, out_path)


def fit_and_save_distributions_donor(genes, cell_types, donor, train_adata, out_path):
    if os.path.exists(out_path):
        return

    donor_dists = {}
    for cell_type in cell_types:
        print(f"Cell type {cell_type}")
        donor_dists[int(cell_type)] = {}
        for gene in genes:
            cell_type_mask = train_adata.obs.cell_type == cell_type
            donor_mask = train_adata.obs.individual == donor
            counts = train_adata[cell_type_mask & donor_mask, gene].X.toarray().flatten()
            if len(counts) == 0 or counts.mean() == 0:
                continue
            try:
                dist = fit_counts_model(counts)
                donor_dists[cell_type][gene] = dist
            except Exception as e:
                continue

    with open(out_path, "w+") as f:
        json.dump(donor_dists, f)


def main(h5ad, hvg_file, splits_dir):
    model_adata = sc.read_h5ad(h5ad)
    hvg_raw = pd.read_csv(hvg_file)
    hvg_mask = hvg_raw["highly_variable"]
    genes = model_adata.var.index[hvg_mask]
    cell_types = model_adata.obs.cell_type.unique()

    num_donors = len(model_adata.obs.individual.unique())

    for i, donor in enumerate(model_adata.obs.individual.unique()):
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")
        print(f"{time_string}: Donor {i} / {num_donors}")

        out_path = os.path.join("donor_dists", f"{donor}.json")
        try:
            fit_and_save_distributions_donor(genes, cell_types, donor, model_adata, out_path)
        except RuntimeError:
            pass

    for i in range(1, 4):
        synth = sc.read_h5ad(f"{splits_dir}/synthetic_{i}.h5ad")
        for j, gene in enumerate(genes):
            print(f"Gene {j} / {len(genes)}")
            for cell_type in cell_types:
                out_path = os.path.join("synth_dist", f"synth_{i}_{gene}_{cell_type}.json")
                try:
                    fit_and_save_distribution(gene, cell_type, synth, out_path)
                except RuntimeError:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two flagged command-line arguments.")

    parser.add_argument("--genes", required=True, help="CSV with boolean 'highly_variable' column, genes index")
    parser.add_argument("--h5ad", required=True, help="Anndata containing expression for all donors")
    parser.add_argument("--splits", required=True, help="Dir with files synth_{i}.h5ad and donors_{i}.csv for each split of data")
    args = parser.parse_args()

    main(args.h5ad, args.genes, args.splits)
