# #%%
import os
import sys
import time
import pickle
from typing import Tuple, Optional, Any
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
import random

# data
import numpy as np
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
from matplotlib.colors import ListedColormap
# data workflows
import tqdm
import scipy
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold

# #%%
DPI = 150
SAVE_DIR="res"
COLORS = ["#e78284", "#a6d189", "#b4befe"]
MY_COLORS = ["#f38ba8", "#cba6f7"]
MY_CMAP = ListedColormap(MY_COLORS)
EDGECOLOR = "#11111b"
STYLES = ["o", "s", "^"]
# MAKE_16x16 = True
MAKE_16x16 = False

# #%%

class Plotter:
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple[int, int] = (6, 6)) -> None:
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.nrows = nrows
        self.ncols = ncols
        self.size = self.nrows * self.ncols
        
        if nrows == 1 and ncols == 1:
            self.axes = np.array([self.axes])
        elif nrows == 1 or ncols == 1:
            self.axes = self.axes.reshape(-1)
        else:
            self.axes = self.axes.flatten()

        self.position = None
    
    def __getitem__(self, index: int):
        return self._get_axes_by_index(index)
    
    def __getattr__(self, name: str):
        if hasattr(mpl_axes.Axes, name):
            def dynamic_method(*args, 
                             idx: Optional[int] = None,
                             irow: Optional[int] = None,
                             icol: Optional[int] = None,
                             **kwargs):
                ax = self._get_axis(idx, irow, icol) if self.position is None else self.position
                method = getattr(ax, name)
                return method(*args, **kwargs)
            
            return dynamic_method
        
        raise AttributeError(f"\"{type(self).__name__}\" object has no attribute \"{name}\"")
    
    def _get_axes_by_index(self, idx: int = 0) -> Any:
        return self.axes[idx]
    
    def _get_axes_by_coords(self, irow: int = 0, icol: int = 0) -> Any:
        idx = irow * self.ncols + icol
        return self.axes[idx]

    def _get_axis(self,
                  idx: Optional[int] = None,
                  irow: Optional[int] = None,
                  icol: Optional[int] = None) -> Any:
        if idx is not None:
            return self._get_axes_by_index(idx)
        elif irow is not None and icol is not None:
            return self._get_axes_by_coords(irow, icol)
        else:
            raise ValueError("ERROR: Wrong indexation!")
    
    def set_position(self,
                     idx: Optional[int] = None,
                     irow: Optional[int] = None,
                     icol: Optional[int] = None) -> None:
        
        self.position = self._get_axis(idx, irow, icol)
    
    def del_position(self) -> None:
        self.position = None
    
    def dataset_visual(self, X, y, colors, styles, edgecolor, alpha=0.6) -> None:
        if self.size != len(X.columns) ** 2:
            raise ValueError("ERROR: Features count not matched with figure size!")
        
        columns = X.columns
        classes = y.unique()
        for i, rowname in enumerate(columns):
            for j, colname in enumerate(columns):
                self.set_position(irow=i, icol=j)
                
                for k, class_value in enumerate(classes):
                    class_mask = y == class_value
                    if i == j:
                        local_data = X[class_mask][colname]
                        kde = scipy.stats.gaussian_kde(local_data)
                        x = np.linspace(local_data.min(), local_data.max(), 100)
                        self.fill_between(x, np.zeros_like(x), kde(x), color=colors[k], alpha=alpha, label=f"class \"{class_value}\"")
                        continue

                    self.scatter(X[class_mask][colname], X[class_mask][rowname], marker=styles[k], color=colors[k], edgecolor=edgecolor, alpha=alpha, label=f"class \"{class_value}\"")
                
                if i == j:
                    self.labels("value", "count", colname)
                else:
                    self.labels(colname, rowname, "")
                
                self.grid(True, alpha=0.3)
                self.legend()
        
    def labels(self,
               xlabel: str,
               ylabel: str,
               title: str,
               idx: Optional[int] = None,
               irow: Optional[int] = None,
               icol: Optional[int] = None) -> Any:

        ax = self._get_axis(idx, irow, icol) if self.position is None else self.position
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax

    def tight_layout(self) -> None:
        self.fig.tight_layout()
    
    def save(self, path: str, dpi: int = 300, **kwargs) -> None:
        self.fig.savefig(path, dpi=dpi, **kwargs)
    
    def show(self) -> None:
        self.fig.show()


def cache_data(cache_file):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_file):
                print(f"\033[90m> Loading from cache: {cache_file}\033[0m")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                print("\033[90m> Generating a new object...\033[0m")
                obj = func(*args, **kwargs)
                with open(cache_file, "wb") as f:
                    pickle.dump(obj, f)
                print(f"\033[90m> Object loaded to cache: {cache_file}\033[0m")
                return obj
        return wrapper
    return decorator


@dataclass
class TimerResult:
    elapsed: float
    start: float
    end: float


@contextmanager
def timer():
    start = time.perf_counter()
    result = TimerResult(elapsed=0, start=start, end=0)
    
    yield result 
    
    result.end = time.perf_counter()
    result.elapsed = result.end - result.start


def select_unique(arr, n):
    unique_sorted = np.unique(arr)
    
    if len(unique_sorted) <= n:
        return unique_sorted
    
    indices = np.linspace(0, len(unique_sorted)-1, n, dtype=int)
    return indices


# #%%
# =============================================================================
# ЗАДАНИЕ 2.1 - Генерация датасета df1
# =============================================================================
print("=== ЗАДАНИЕ 2.1 - Генерация датасета df1 ===")

np.random.seed(42)
n_samples = 1000
n_features = 16
n_classes = 3

COLUMNS = [f"feature_{i+1}" for i in range(n_features)]


@cache_data("cache/df1_main.pkl")
def create_dataframe(n_samples, n_features, n_classes):
    X1, y1 = make_blobs(n_samples=n_samples * (n_classes - 1), 
                        n_features=n_features, 
                        centers=2,
                        cluster_std=2.5,
                        random_state=42)
    X2, y2 = make_blobs(n_samples=n_samples, 
                        n_features=n_features, 
                        centers=1,
                        cluster_std=1.6,
                        center_box=(10.0, 15.0),
                        random_state=42)

    X = np.vstack([X1, X2])
    y = np.hstack([y1, np.full(y2.shape, 2)])

    df = pd.DataFrame(X, columns=COLUMNS)
    df["target"] = y

    return df


df1 = create_dataframe(n_samples, n_features, n_classes)

print(f"Размерность df1: {df1.shape}")
print(f"Количество классов: {len(df1["target"].unique())}")
print("Количество объектов в каждом классе:")
print(df1["target"].value_counts().sort_index())
print(df1.head())

# #%%
X = df1[COLUMNS]
y = df1["target"]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

plotter = Plotter(nrows=1, ncols=1, figsize=(10, 8))
plotter.set_position(idx=0)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

for class_value in range(n_classes):
    mask = y == class_value
    plotter.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.5, marker=STYLES[class_value], color=COLORS[class_value], edgecolor=EDGECOLOR, label=f"class_{class_value}")

plotter.labels("PCA axes 1", "PCA axes 2", "PCA visualization")
plotter.grid(True, alpha=0.3)
plotter.legend()
plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/data_PCA.png", dpi=DPI)

# #%%
plotter = Plotter(nrows=4, ncols=4, figsize=(24, 24))

for i, col in enumerate(COLUMNS):
    plotter.set_position(idx=i)

    for class_value in range(n_classes):
        mask = y == class_value
        plotter.hist(df1[mask][col], bins=30, color=COLORS[class_value], alpha=0.5, edgecolor=EDGECOLOR, label=f"class_{class_value}")
    plotter.labels("value", "count", col)
    plotter.grid(True, alpha=0.3)
    plotter.legend()

plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/df1_hists.png", dpi=DPI)

# #%%
# =============================================================================
# ЗАДАНИЕ 2.2 - Создание датасетов с повторенными объектами
# =============================================================================
print("\n=== ЗАДАНИЕ 2.2 - Создание датасетов с повторенными объектами ===")

CLASS_TO_REPEAT = 1
repetition_factors = [2, 5, 10, 20, 50, 100, 1000, 10000]

datasets = {"df1": df1}

for factor in repetition_factors:
    df_name = f"df{factor}"
    
    @cache_data(f"cache/{df_name}.pkl")
    def generate_copy(base_df):
        new_df = base_df.copy()
        class_samples = df1[df1["target"] == CLASS_TO_REPEAT]
        repeated_samples = pd.concat([class_samples] * (factor - 1), ignore_index=True)
        new_df = pd.concat([new_df, repeated_samples], ignore_index=True)
        
        return new_df
    
    new_df = generate_copy(df1)
    datasets[df_name] = new_df
    print(f"Getted {df_name}: {new_df.shape} objects")

DATASET_NAMES = list(datasets.keys())

# #%%

if MAKE_16x16:
    plotter = Plotter(nrows=16, ncols=16, figsize=(50, 50))

    plotter.dataset_visual(df1[COLUMNS], df1["target"], COLORS, STYLES, EDGECOLOR, alpha=0.6)

    plotter.tight_layout()
    plotter.save("res/df1_16x16.png", dpi=200)

# #%%

def fit_model(name, model, X, y):
    with timer() as t:
        @cache_data(name)
        def func(X, y):
            l_model = model
            l_model.fit(X, y)
            
            return l_model
        
        fitted_model = func(X, y)

    return fitted_model, t.elapsed


def filter_data(df, classes, features):
    df_filtered = df[df["target"].isin(classes)].copy()
    df_filtered["target"] = pd.factorize(df_filtered["target"])[0]
    X, y = df_filtered[features].values, df_filtered["target"].values
    
    return df_filtered, X, y


def calc_mass_points(df, name, features):
    centers = {}
    classes = df["target"].unique()
    for class_label in classes:
        class_data = df[df["target"] == class_label][features]
        center = class_data.mean().values
        centers[class_label] = center
    
    overall_center = df[features].mean().values
    
    midpoint = (centers[classes[0]] + centers[classes[1]]) / 2
    
    return {
        "dataset": name,
        "center_class_0": centers[classes[0]],
        "center_class_1": centers[classes[1]],
        "overall_center": overall_center,
        "midpoint": midpoint
    }


def ROC_PR_calc(model, X, y, roc=True, pr=True, no_proj=False):
    prediction = model.predict_proba(X)[:, 1]
    if no_proj:
        projections = prediction
    else:
        projections = model.transform(X).ravel()
    size = len(projections)

    res = {"prediction": prediction,
           "projections": projections,
           "size": size}

    if roc:
        fpr, tpr, _ = roc_curve(y, prediction)
        aucroc = auc(fpr, tpr)
        
        res["fpr"] = fpr
        res["tpr"] = tpr
        res["aucroc"] = aucroc
    if pr:
        precision, recall, _ = precision_recall_curve(y, prediction)
        precision_orig_interp = np.interp(recall, recall[::-1], precision[::-1], left=1.0, right=0.0)
        auprc = auc(recall, precision)
        
        res["recall"] = recall
        res["precision"] = precision_orig_interp
        res["auprc"] = auprc
    
    return res


def ROC_PR_calc_with_area(model, X, y, roc=True, pr=True, n_bootstraps=1000, confidence_level=0.95, no_proj=False):
    data = ROC_PR_calc(model, X, y, roc, pr, no_proj=no_proj)

    roc_scores = []
    pr_scores = []
    tpr_matrix = []
    precision_matrix = []

    for _ in tqdm.tqdm(range(n_bootstraps), desc=f"ROC and PR {confidence_level * 100}% CI calculating", unit="Bootstrap"):
        idx = np.random.choice(data["size"], data["size"], replace=True)
        if roc:
            fpr_boot, tpr_boot, _ = roc_curve(y[idx], data["projections"][idx])
            roc_scores.append(auc(fpr_boot, tpr_boot))
            tpr_interp = np.interp(data["fpr"], fpr_boot, tpr_boot, left=0, right=1)
            tpr_matrix.append(tpr_interp)
        if pr:
            precision_boot, recall_boot, _ = precision_recall_curve(y[idx], data["projections"][idx])
            pr_scores.append(auc(recall_boot, precision_boot))
            precision_interp = np.interp(data["recall"], recall_boot[::-1], precision_boot[::-1], left=1, right=0)
            precision_matrix.append(precision_interp)
    
    if roc:
        tpr_matrix = np.array(tpr_matrix)
    if pr:
        precision_matrix = np.array(precision_matrix)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    res = data
    
    if roc:
        tpr_lower = np.percentile(tpr_matrix, lower_percentile, axis=0)
        tpr_upper = np.percentile(tpr_matrix, upper_percentile, axis=0)
        tpr_mean = np.mean(tpr_matrix, axis=0) 
        tpr_area = np.trapezoid(tpr_upper - tpr_lower, data["fpr"])
        
        res["roc_score_mean"] = np.mean(roc_scores)
        res["tpr_lower"] = tpr_lower
        res["tpr_upper"] = tpr_upper
        res["tpr_mean"] = tpr_mean
        res["tpr_area"] = tpr_area

    if pr:
        precision_lower = np.percentile(precision_matrix, lower_percentile, axis=0)
        precision_upper = np.percentile(precision_matrix, upper_percentile, axis=0)
        precision_mean = np.mean(precision_matrix, axis=0) 
        precision_area = np.trapezoid(precision_upper - precision_lower, data["recall"])
        
        res["pr_score_mean"] = np.mean(pr_scores)
        res["precision_lower"] = precision_lower
        res["precision_upper"] = precision_upper
        res["precision_mean"] = precision_mean
        res["precision_area"] = precision_area
    
    return res


def draw_curve(plotter, idx, ox, oy_origin, oy_mean, oy_lower, oy_upper, score, score_mean, area, conf_level, name, labels, line=([0, 1], [0, 1])):
    plotter.set_position(idx=idx)

    plotter.plot(*line, color="red", alpha=0.5, linestyle="--")
    plotter.fill_between(ox, oy_lower, oy_upper, color="green", alpha=0.3, label=f"{name} CI for {conf_level * 100}% ({area:.6f})")
    plotter.plot(ox, oy_mean, color="blue", linewidth=2, label=f"{name}-curve mean ({score_mean:.6f})")
    plotter.plot(ox, oy_origin, color="red", linewidth=1, label=f"{name}-curve origin ({score:.6f})")
    plotter.grid(True, alpha=0.3)
    plotter.legend()
    plotter.labels(*labels)


def draw_model(plotter, idx, size, model, X, y, centers_info, labels, scatter_max=1000):
    plotter.set_position(idx=idx)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, size),
                         np.linspace(y_min, y_max, size))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plotter.contourf(xx, yy, Z, alpha=0.3, cmap=MY_CMAP)
    selected_indexes = select_unique(X[:, 0], scatter_max)
    plotter.scatter(X[:, 0][selected_indexes], X[:, 1][selected_indexes], c=y[selected_indexes], 
                    cmap=MY_CMAP, edgecolors=EDGECOLOR, s=30)

    plotter.plot([centers_info["center_class_0"][0], centers_info["center_class_1"][0]],
                 [centers_info["center_class_0"][1], centers_info["center_class_1"][1]], "k--", linewidth=3)
    plotter.scatter(centers_info["center_class_0"][0], centers_info["center_class_0"][1], 
                   c="red", marker="X", s=250, label="Center Class 0", alpha=0.8, edgecolors=EDGECOLOR)
    plotter.scatter(centers_info["center_class_1"][0], centers_info["center_class_1"][1], 
                   c="blue", marker="X", s=250, label="Center Class 1", alpha=0.8, edgecolors=EDGECOLOR)
    plotter.scatter(centers_info["overall_center"][0], centers_info["overall_center"][1], 
                   c="green", marker="*", s=300, label="Overall Center", alpha=0.8, edgecolors=EDGECOLOR)
    plotter.scatter(centers_info["midpoint"][0], centers_info["midpoint"][1], 
                   c="purple", marker="D", s=150, label="Midpoint", alpha=0.8, edgecolors=EDGECOLOR)

    plotter.labels(*labels)
    plotter.legend()


# #%%

def make_stats_for_model(plotter, model, name, datasets, features, classes, n_bootstraps=(1000, 10), confidence_level=0.95, no_proj=False):
    centers_data = []
    lda_times = {}
    for i, dataset in enumerate(DATASET_NAMES):
        if dataset not in datasets:
            continue
        
        df = datasets[dataset]
        
        df_filtered, X_plot, y_plot = filter_data(df, classes, features)

        centers_data.append(calc_mass_points(df_filtered, dataset, features))
        
        model, time_elapsed = fit_model(f"cache/{name}_{classes[0]}_{classes[1]}_{features[0]}_{features[1]}_{dataset}.pkl",
                                        model, X_plot, y_plot)

        lda_times[dataset] = time_elapsed
        
        roc_pr_stat = ROC_PR_calc_with_area(model, X_plot, y_plot, n_bootstraps=(n_bootstraps[0] if len(y_plot) < 1000000 else n_bootstraps[1]), no_proj=no_proj)

        plotter.set_position(idx=i * 3 + 1)
        
        draw_curve(plotter, i * 3 + 1,
                   roc_pr_stat["fpr"], roc_pr_stat["tpr"], roc_pr_stat["tpr_mean"],
                   roc_pr_stat["tpr_lower"], roc_pr_stat["tpr_upper"],
                   roc_pr_stat["aucroc"], roc_pr_stat["roc_score_mean"],
                   roc_pr_stat["tpr_area"],
                   confidence_level, "ROC", ("FPR", "TPR", f"ROC-curve for {dataset}"))

        draw_curve(plotter, i * 3 + 2,
                   roc_pr_stat["recall"], roc_pr_stat["precision"], roc_pr_stat["precision_mean"],
                   roc_pr_stat["precision_lower"], roc_pr_stat["precision_upper"],
                   roc_pr_stat["auprc"], roc_pr_stat["pr_score_mean"],
                   roc_pr_stat["precision_area"],
                   confidence_level, "PR", ("recall", "precision", f"PR-curve for {dataset}"), line=([0, 1], [1, 0]))
        plotter.invert_xaxis()

        draw_model(plotter, i * 3, 500, model, X_plot, y_plot, centers_data[-1], (features[0], features[1], dataset))

    centers_df = pd.DataFrame(centers_data)
    print("Mass centers:")
    print(centers_df)


# #%%
selected_classes = [0, 1]
# selected_features = [COLUMNS[0], COLUMNS[3]]
selected_features = [COLUMNS[6], COLUMNS[8]]
# selected_features = [COLUMNS[5], COLUMNS[15]]

plotter = Plotter(nrows=len(datasets), ncols=3, figsize=(16, 32))

make_stats_for_model(plotter, LinearDiscriminantAnalysis(), "lda", datasets, selected_features, selected_classes)

plotter.tight_layout()
plotter.save("res/LDA_ROC_PR_1.png", dpi=DPI)

# #%%

selected_classes = [1, 2]
# selected_features = [COLUMNS[0], COLUMNS[3]]
selected_features = [COLUMNS[6], COLUMNS[8]]
# selected_features = [COLUMNS[5], COLUMNS[15]]

plotter = Plotter(nrows=len(datasets), ncols=3, figsize=(16, 32))

make_stats_for_model(plotter, LinearDiscriminantAnalysis(), "lda", datasets, selected_features, selected_classes)

plotter.tight_layout()
plotter.save("res/LDA_ROC_PR_2.png", dpi=DPI)

# #%%

confidence_level = 0.95
n_bootstraps = (1000, 10)
selected_classes = [0, 1]

plotter = Plotter(nrows=4, ncols=2, figsize=(16, 24))

df = datasets["df1"]
    
for i, current_n_features in enumerate([2, 4, 8, 16]):
    selected_features = random.sample(COLUMNS, current_n_features)
    
    df_filtered, X_plot, y_plot = filter_data(df, selected_classes, selected_features)
    
    model, time_elapsed = fit_model(f"cache/lda_many_{selected_classes[0]}_{selected_classes[1]}_{current_n_features}.pkl",
                                    LinearDiscriminantAnalysis(), X_plot, y_plot)

    roc_pr_stat = ROC_PR_calc_with_area(model, X_plot, y_plot, n_bootstraps=(n_bootstraps[0] if len(y_plot) < 1000000 else n_bootstraps[1]))

    draw_curve(plotter, i * 2,
               roc_pr_stat["fpr"], roc_pr_stat["tpr"], roc_pr_stat["tpr_mean"],
               roc_pr_stat["tpr_lower"], roc_pr_stat["tpr_upper"],
               roc_pr_stat["aucroc"], roc_pr_stat["roc_score_mean"],
               roc_pr_stat["tpr_area"],
               confidence_level, "ROC", ("FPR", "TPR", f"ROC-curve for {current_n_features} features"))

    draw_curve(plotter, i * 2 + 1,
               roc_pr_stat["recall"], roc_pr_stat["precision"], roc_pr_stat["precision_mean"],
               roc_pr_stat["precision_lower"], roc_pr_stat["precision_upper"],
               roc_pr_stat["auprc"], roc_pr_stat["pr_score_mean"],
               roc_pr_stat["precision_area"],
               confidence_level, "PR", ("recall", "precision", f"PR-curve for {current_n_features} features"), line=([0, 1], [1, 0]))
    plotter.invert_xaxis()

plotter.tight_layout()
plotter.save("res/ROC_PR_many_features.png", dpi=DPI)

# #%%
selected_classes = [0, 1]
# selected_features = [COLUMNS[0], COLUMNS[3]]
selected_features = [COLUMNS[6], COLUMNS[8]]
# selected_features = [COLUMNS[5], COLUMNS[15]]

plotter = Plotter(nrows=len(datasets), ncols=3, figsize=(16, 32))

make_stats_for_model(plotter, QuadraticDiscriminantAnalysis(), "qda", datasets, selected_features, selected_classes, no_proj=True)

plotter.tight_layout()
plotter.save("res/QDA_ROC_PR.png", dpi=DPI)

sys.exit(0)
