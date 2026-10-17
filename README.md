# Predicting Inorganic Fertilizer Adoption on Rwandan Farm Plots

Machine Learning module, summative project: a comparison of traditional machine learning (scikit-learn) and deep learning (TensorFlow/Keras) for a binary classification task built on Rwanda's national agricultural survey.

> **Links:** Report: _add link_ · Google Drive (models): _add link_ · Video: _add link_

## Problem

<!-- Write 3–4 sentences in your own words: why fertilizer adoption matters for Rwanda,
     who would use this model (e.g., RAB extension officers, MINAGRI subsidy programmes),
     and what the model predicts. -->

**Task:** given the location, farmer and practice characteristics of a farm plot in a season, predict whether inorganic (mineral) fertilizer was applied to the plot.

## Dataset

- **Source:** National Institute of Statistics of Rwanda (NISR), *Seasonal Agricultural Survey 2020*, National Data Archive.
- **Modules used:** Fertilizers & Pesticides, Agricultural Practice, and Crop Production, for seasons A and B.
- **Size:** about 150,000 raw rows across these six files, which reduce to **33,228 plot-seasons** from **16,854 farmers** in all 30 districts.
- **Target balance:** 27.6% of plots use inorganic fertilizer and 72.4% do not.

**Data access:** the microdata is licensed by NISR and may not be redistributed, so it is **not** in this repository. To reproduce the results:
1. Request access to *Rwanda Seasonal Agricultural Survey 2020* at <https://microdata.statistics.gov.rw>.
2. Unzip the download so that the Stata files are in `data/raw/Datasets/STATA/`.

## Repository structure

```
├── notebooks/
│   ├── 01_data_preparation_and_eda.ipynb   # merging, leakage removal, EDA, grouped split
│   ├── 02_traditional_ml.ipynb             # LR, RF, Gradient Boosting, SVM: tuning, ablation, evaluation
│   └── 03_deep_learning.ipynb              # Keras (Sequential + Functional) with tf.data, ML vs DL comparison
├── src/
│   ├── features.py                         # feature engineering + preprocessing pipeline
│   └── evaluation.py                       # metrics and plotting helpers
├── figures/                                # plots exported by the notebooks
├── results/                                # experiment tables and test metrics (CSV/JSON)
├── models/                                 # gitignored; saved models go to Google Drive
├── data/                                   # gitignored; raw and processed data live here locally
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone <your-repo-url>
cd rwanda-fertilizer-ml
pip install -r requirements.txt
# place the NISR data in data/raw/Datasets/STATA/ (see above)
jupyter notebook notebooks/01_data_preparation_and_eda.ipynb
```

Run the notebooks in order. Notebooks 2 and 3 each take about 15–30 minutes on a laptop CPU (no GPU needed); most of the time goes on hyperparameter search. Notebook 1 writes `data/processed/plots_model_table.parquet` and `feature_spec.json`, which notebooks 2 and 3 read. All randomness uses `SEED = 42`.

## Key preprocessing decisions

- **One row per plot-season.** The fertilizer and crop modules have several rows per plot, which are combined into one.
- **Leakage removed.** Farmer-level "any plot used inorganic fertilizer" (q3.7), fertilizer type, quantity and price (q3.8, q3.10–3.15), micronutrients, pesticide use, and all harvest and sales outcomes are excluded.
- **Farmer-grouped split** (70/15/15). No farmer appears in more than one of train, validation and test.

## Results

Test set: 5,045 plots from farmers never seen in training. Thresholds were tuned on the validation set.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.878 | 0.719 | 0.695 | 0.623 | 0.784 |
| Random Forest | 0.890 | 0.763 | 0.709 | 0.686 | 0.733 |
| SVM (RBF) | 0.886 | 0.725 | 0.714 | 0.640 | 0.808 |
| Gradient Boosting (best traditional ML) | 0.892 | 0.766 | 0.712 | 0.677 | 0.751 |
| **Neural network, 128-64 MLP + BatchNorm + dropout (best DL)** | **0.897** | **0.768** | **0.716** | **0.703** | 0.729 |

The network's ROC-AUC advantage over Gradient Boosting is +0.005 (95% paired-bootstrap CI +0.002 to +0.009).

<!-- Add 2–3 sentences in your own words: what this means and which model you recommend. -->

## Saved models (Google Drive)

| File | Contents | How to load |
|---|---|---|
| `best_traditional_ml.joblib` | Full scikit-learn pipeline (preprocessing + Gradient Boosting) + decision threshold | `joblib.load(path)["pipeline"].predict_proba(df)` |
| `best_deep_learning.keras` | Keras network (native format) | `keras.models.load_model(path)` |
| `best_deep_learning.h5` | Same network, legacy HDF5 format | `keras.models.load_model(path)` |
| `dl_preprocessor.joblib` | Fitted `ColumnTransformer` the network expects + threshold | `joblib.load(path)["preprocessor"].transform(df)` |

## Citation

National Institute of Statistics of Rwanda (NISR), *Seasonal Agricultural Survey 2020*. Kigali, Rwanda. Ref. RWA_2020_RSAS_v01_EN_M_v01_A_OCS.
