# #%%
import os
import sys
import time
import pickle
from typing import Tuple, Optional, Any
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
from joblib import Parallel, delayed
import multiprocessing

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
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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
    X, y = df_filtered[features].values, df_filtered["target"].values
    
    return df_filtered, X, y


def calc_mass_points(df, name, classes, features):
    centers = {}
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


def ROC_PR_calc(model, X, y, roc=True, pr=True):
    prediction = model.predict_proba(X)[:, 1]
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


def ROC_PR_calc_with_area(model, X, y, roc=True, pr=True, n_bootstraps=1000, confidence_level=0.95):
    data = ROC_PR_calc(model, X, y, roc, pr)

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

confidence_level = 0.95
n_bootstraps = 1000
selected_classes = [0, 1]
# selected_features = [COLUMNS[0], COLUMNS[3]]
selected_features = [COLUMNS[5], COLUMNS[15]]

plotter = Plotter(nrows=9, ncols=3, figsize=(16, 32))

centers_data = []
lda_times = {}
for i, dataset in enumerate(DATASET_NAMES):
    if dataset not in datasets:
        continue
    
    df = datasets[dataset]
    
    df_filtered, X_plot, y_plot = filter_data(df, selected_classes, selected_features)

    centers_data.append(calc_mass_points(df_filtered, dataset, selected_classes, selected_features))
    
    model, time_elapsed = fit_model(f"cache/lda_{selected_classes[0]}_{selected_classes[1]}_{selected_features[0]}_{selected_features[1]}_{dataset}.pkl",
                      LinearDiscriminantAnalysis(), X_plot, y_plot)

    lda_times[dataset] = time_elapsed
    
    roc_pr_stat = ROC_PR_calc_with_area(model, X_plot, y_plot, n_bootstraps=(n_bootstraps if len(y_plot) < 1000000 else 10))

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

    draw_model(plotter, i * 3, 500, model, X_plot, y_plot, centers_data[-1], (selected_features[0], selected_features[1], dataset))

centers_df = pd.DataFrame(centers_data)
print("Mass centers:")
print(centers_df)

plotter.tight_layout()
plotter.save("res/LDA_ROC.png", dpi=DPI)

sys.exit(0)
# #%%

# =============================================================================
# ЗАДАНИЕ 2.8 - Кросс-валидация для df10
# =============================================================================
print("\n=== ЗАДАНИЕ 2.8 - Кросс-валидация для df10 ===")

df10 = datasets["df10"]
df_filtered = df10[df10["target"].isin(selected_classes)].copy()

X_data = df_filtered[selected_features].values
y_data = df_filtered["target"].values
y_binary = (y_data == CLASS_TO_REPEAT).astype(int)

# Параметры кросс-валидации
cv_folds = [3, 5, 10, 20, 50, 100]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Для ROC кривых
for n_folds in cv_folds:
    tprs = []
    base_fpr = np.linspace(0, 1, 100)
    auc_scores = []
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, test_idx in cv.split(X_data, y_binary):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        
        y_score = lda.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        
        auc_scores.append(auc(fpr, tpr))
    
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    
    mean_auc = np.mean(auc_scores)
    
    ax1.plot(base_fpr, mean_tpr, label=f"{n_folds}-fold (AUC={mean_auc:.3f})")
    ax1.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.1)

ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC кривые с кросс-валидацией (df10)")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Для PR кривых
for n_folds in cv_folds:
    precisions_list = []
    base_recall = np.linspace(0, 1, 100)
    auprc_scores = []
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, test_idx in cv.split(X_data, y_binary):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        
        y_score = lda.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
        precisions_list.append(precision_interp)
        
        auprc_scores.append(auc(recall, precision))
    
    precisions_list = np.array(precisions_list)
    mean_precision = precisions_list.mean(axis=0)
    std_precision = precisions_list.std(axis=0)
    
    precision_upper = np.minimum(mean_precision + 1.96 * std_precision, 1)
    precision_lower = np.maximum(mean_precision - 1.96 * std_precision, 0)
    
    mean_auprc = np.mean(auprc_scores)
    
    ax2.plot(base_recall, mean_precision, label=f"{n_folds}-fold (AUPRC={mean_auprc:.3f})")
    ax2.fill_between(base_recall, precision_lower, precision_upper, alpha=0.1)

baseline = np.mean(y_binary)
ax2.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline = {baseline:.3f}")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("PR кривые с кросс-валидацией (df10)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cross_validation_curves.png", dpi=300, bbox_inches="tight")
#plt.show()

'''
# =============================================================================
# ЗАДАНИЕ 2.9 - LDA с разным количеством признаков
# =============================================================================
print("\n=== ЗАДАНИЕ 2.9 - LDA с разным количеством признаков ===")

feature_counts = [2, 4, 8, 16]
df1_data = datasets["df1"]
df_filtered = df1_data[df1_data["target"].isin(selected_classes)].copy()

y_binary = (df_filtered["target"] == CLASS_TO_REPEAT).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, n_features in enumerate(feature_counts):
    if i >= len(axes):
        break
        
    # Выбираем первые n_features признаков
    selected_feature_subset = [f"feature_{j+1}" for j in range(n_features)]
    X_data = df_filtered[selected_feature_subset].values
    
    # Бутстреп ROC кривых
    lda = LinearDiscriminantAnalysis()
    fpr, mean_tpr, tpr_lower, tpr_upper, auc_scores = bootstrap_roc_curve(
        X_data, y_binary, lda, n_bootstrap=100, target_class=1)
    
    mean_auc = np.mean(auc_scores)
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    
    # Визуализация
    axes[i].plot(fpr, mean_tpr, color="b", label=f"ROC (AUC = {mean_auc:.3f})")
    axes[i].fill_between(fpr, tpr_lower, tpr_upper, color="grey", alpha=0.3,
                        label="95% доверительный интервал")
    axes[i].plot([0, 1], [0, 1], "k--", alpha=0.5)
    
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel("False Positive Rate")
    axes[i].set_ylabel("True Positive Rate")
    axes[i].set_title(f"LDA с {n_features} признаками\nAUC: {mean_auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    axes[i].legend(loc="lower right")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lda_different_features.png", dpi=300, bbox_inches="tight")
#plt.show()

# =============================================================================
# ЗАДАНИЕ 2.10* - QDA анализ
# =============================================================================
print("\n=== ЗАДАНИЕ 2.10* - QDA анализ ===")

# Визуализация QDA для разных датасетов
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

qda_times = {}

for i, df_name in enumerate(DATASET_NAMES):
    if df_name not in datasets or i >= len(axes):
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    X_plot = df_filtered[selected_features].values
    y_plot = df_filtered["target"].values
    
    # Время начала обучения QDA
    start_time = time.time()
    
    # Обучаем QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_plot, y_plot)
    
    # Время окончания обучения
    end_time = time.time()
    qda_times[df_name] = end_time - start_time
    
    # Создаем сетку для построения решающей границы
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Предсказываем для всех точек сетки
    Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Визуализация
    contour = axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = axes[i].scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, 
                             cmap=plt.cm.RdYlBu, edgecolors="black", s=30)
    
    axes[i].set_title(f"QDA - {df_name}\nВремя: {qda_times[df_name]:.4f}с", fontsize=12)
    axes[i].set_xlabel(selected_features[0])
    axes[i].set_ylabel(selected_features[1])

plt.tight_layout()
plt.savefig("qda_decision_boundaries.png", dpi=300, bbox_inches="tight")
#plt.show()

print("Время выполнения QDA для разных датасетов:")
for df_name, t in qda_times.items():
    print(f"{df_name}: {t:.4f} секунд")

# ROC кривые для QDA
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

qda_roc_data = {}

for i, df_name in enumerate(DATASET_NAMES):
    if df_name not in datasets or i >= len(axes):
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    X_data = df_filtered[selected_features].values
    y_data = df_filtered["target"].values
    y_binary = (y_data == CLASS_TO_REPEAT).astype(int)
    
    # Бутстреп ROC кривых для QDA
    qda = QuadraticDiscriminantAnalysis()
    fpr, mean_tpr, tpr_lower, tpr_upper, auc_scores = bootstrap_roc_curve(
        X_data, y_binary, qda, n_bootstrap=100, target_class=1)
    
    mean_auc = np.mean(auc_scores)
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    
    qda_roc_data[df_name] = {
        "mean_auc": mean_auc,
        "auc_ci": auc_ci
    }
    
    # Визуализация ROC кривой
    axes[i].plot(fpr, mean_tpr, color="b", label=f"QDA ROC (AUC = {mean_auc:.3f})")
    axes[i].fill_between(fpr, tpr_lower, tpr_upper, color="grey", alpha=0.3,
                        label="95% доверительный интервал")
    axes[i].plot([0, 1], [0, 1], "k--", alpha=0.5)
    
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel("False Positive Rate")
    axes[i].set_ylabel("True Positive Rate")
    axes[i].set_title(f"QDA ROC - {df_name}\nAUC: {mean_auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    axes[i].legend(loc="lower right")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qda_roc_curves.png", dpi=300, bbox_inches="tight")
#plt.show()

print("Сравнение LDA и QDA:")
print("Датасет\t\tLDA AUC\t\tQDA AUC")
for df_name in DATASET_NAMES:
    if df_name in roc_auc_data and df_name in qda_roc_data:
        lda_auc = roc_auc_data[df_name]["mean_auc"]
        qda_auc = qda_roc_data[df_name]["mean_auc"]
        print(f"{df_name}\t\t{lda_auc:.3f}\t\t{qda_auc:.3f}")

print("\n=== Анализ завершен ===")
'''
