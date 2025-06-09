import numpy as np
import os
import argparse
import json
import scanpy as sc
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

def kruskal_donor_variability(adata, genes, out_dir, donor_key='individual', ct_key='cell_type', correction='bonferroni'):
    donors = adata.obs[donor_key].unique()
    cts = adata.obs[ct_key].unique()
    results_full = {}

    for ct in cts:
        pvals = []
        for i, gene in enumerate(genes):
            print(f"Gene {i} / {len(genes)}")
            gene_expr = adata[:, gene].X
            if hasattr(gene_expr, "toarray"):
                gene_expr = gene_expr.toarray().flatten()
            else:
                gene_expr = gene_expr.flatten()

            cell_type_mask = adata.obs['cell_type'] == ct
            groups = [gene_expr[(adata.obs[donor_key] == donor) & cell_type_mask] for donor in donors]

            try:
                stat, p = kruskal(*groups)
            except ValueError:
                p = 1.0 
                
            pvals.append(p)

        pvals = np.array(pvals)
        corrected = multipletests(pvals, method=correction)
        pvals_corrected = corrected[1]
        rejected = corrected[0]

        results_full[ct] = pd.DataFrame({
            'gene': genes,
            'pval': pvals,
            'pval_corrected': pvals_corrected,
            'significant': rejected
        }).sort_values('pval_corrected')
        results_full[ct].to_csv(os.path.join(out_dir, f"{ct}.csv"))

    return results_full

def main():
    parser = argparse.ArgumentParser(description="Find donor-variable genes.")

    parser.add_argument("--genes", required=True, help="CSV with boolean 'highly_variable' column, genes index")
    parser.add_argument("--h5ad", required=True, help="Input anndata")
    parser.add_argument("--out_dir", required=True, help="Output path")
    args = parser.parse_args()

    hvg_raw = pd.read_csv(args.genes)
    hvg_mask = hvg_raw["highly_variable"]
    adata = sc.read_h5ad(args.h5ad)
    hvgs = adata[:, hvg_mask].var.index
    #hvgs = adata.var.index
    results = kruskal_donor_variability(adata, hvgs, args.out_dir)
    results.to_csv(args.out)


if __name__ == "__main__":
    main()
