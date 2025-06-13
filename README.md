# Requirements

We assume there is a directory containing synthetic data (named as `synthetic_{i}.h5ad`)
for given splits of donors (listed in `scdesign_attack/donors_{i}.csv`). We assume there is an
H5AD file containing expression level for full donors. We assume there is an HVG file giving
the subset of genes used to train the synthetic data generator. We assume you have run `pip install -r requirements.txt` before running the scripts below.

# Fitting marginal distributions over donors

```
python3 dist.py --genes {hvg_csv} --h5ad {real_data} --splits {splits_dir}
```

# Finding donor-variable genes (optional)

```
python3 kruskal.py --genes {hvg_csv} --h5ad {real_data} --out_dir {out_directory}
```

# Running MIA
Run on all HVGs:
```
python3 mia.py --genes {hvg_csv} --h5ad {real_data} --splits {splits_dir}
```

If you want to run an MIA only on the donor-variable genes identified in the previous step, use
the output directory for that step as the argument to `--dvg_dir` in this step.
```
python3 mia.py --genes {hvg_csv} --h5ad {real_data} --splits {splits_dir} --dvg_dir {dvg_dir}
```

(Syntax is the same for `mia2.py`; this file just runs the version Steven proposed, where we sum over
value of cells' expression values in PMF.)
