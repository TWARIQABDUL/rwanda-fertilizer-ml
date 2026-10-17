"""Feature engineering and preprocessing shared by the traditional-ML and deep-learning notebooks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Engineered farmer-level features (built from features only, never from the target)
FARMER_FEATURES = ["log_farmer_n_plots", "log_farmer_total_area", "plot_share_of_farm"]


def load_model_table(processed_dir: Path):
    """Load the plot-level table and feature lists written by notebook 1."""
    df = pd.read_parquet(processed_dir / "plots_model_table.parquet")
    spec = json.load(open(processed_dir / "feature_spec.json"))
    return df, spec


def add_farmer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Describe each plot in the context of the farmer's whole farm in the same season.

    A plot that is one of many on a large farm is managed differently from the only plot
    of a subsistence household. These features use plot area only (no target information),
    and because the train/val/test split is grouped by farmer, they cannot leak across splits.
    """
    df = df.copy()
    area = np.exp(df["log_plot_area"])
    g = df.groupby(["farmer_id", "season"])
    n_plots = g["plot_uid"].transform("size")
    total_area = area.groupby([df["farmer_id"], df["season"]]).transform("sum")
    df["log_farmer_n_plots"] = np.log(n_plots)
    df["log_farmer_total_area"] = np.log(total_area)
    df["plot_share_of_farm"] = area / total_area
    return df


def build_preprocessor(categorical, numeric, binary):
    """One-hot encode categoricals, impute and scale numerics, pass binary flags through.

    Imputation and scaling statistics are learned inside the pipeline, i.e. from the
    training fold only, so no information leaks from validation/test data.
    """
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    transformers = []
    if categorical: transformers.append(("cat", cat, categorical))
    if numeric:     transformers.append(("num", num, numeric))
    if binary:      transformers.append(("bin", "passthrough", binary))
    return ColumnTransformer(transformers)
