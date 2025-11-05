import os
import json
import hashlib
from pathlib import Path

# import sys
import time
import pickle
from typing import Tuple, Optional, Any, Callable
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# data workflows
import tqdm
import scipy
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold


DPI = 150
SAVE_DIR = "res"
COLORS = ["red", "green", "blue"]
MY_COLORS = ["#f38ba8", "#cba6f7"]
MY_CMAP = ListedColormap(MY_COLORS)
MY_SMOOTH_CMAP = LinearSegmentedColormap.from_list(
    "my_smooth_gradient", MY_COLORS[::-1], N=256
)
EDGECOLOR = "#11111b"
STYLES = ["o", "s", "^"]

CALC_COV = False
CALC_GRID = False


class Plotter:
    def __init__(
        self, nrows: int = 1, ncols: int = 1, figsize: Tuple[int, int] = (6, 6)
    ) -> None:
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

            def dynamic_method(
                *args,
                idx: Optional[int] = None,
                irow: Optional[int] = None,
                icol: Optional[int] = None,
                **kwargs,
            ):
                ax = (
                    self._get_axis(idx, irow, icol)
                    if self.position is None
                    else self.position
                )
                method = getattr(ax, name)
                return method(*args, **kwargs)

            return dynamic_method

        raise AttributeError(
            f'"{type(self).__name__}" object has no attribute "{name}"'
        )

    def _get_axes_by_index(self, idx: int = 0) -> Any:
        return self.axes[idx]

    def _get_axes_by_coords(self, irow: int = 0, icol: int = 0) -> Any:
        idx = irow * self.ncols + icol
        return self.axes[idx]

    def _get_axis(
        self,
        idx: Optional[int] = None,
        irow: Optional[int] = None,
        icol: Optional[int] = None,
    ) -> Any:
        if idx is not None:
            return self._get_axes_by_index(idx)
        elif irow is not None and icol is not None:
            return self._get_axes_by_coords(irow, icol)
        else:
            raise ValueError("ERROR: Wrong indexation!")

    def set_position(
        self,
        idx: Optional[int] = None,
        irow: Optional[int] = None,
        icol: Optional[int] = None,
    ) -> None:
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
                        self.fill_between(
                            x,
                            np.zeros_like(x),
                            kde(x),
                            color=colors[k],
                            alpha=alpha,
                            label=f'class "{class_value}"',
                        )
                        continue

                    self.scatter(
                        X[class_mask][colname],
                        X[class_mask][rowname],
                        marker=styles[k],
                        color=colors[k],
                        edgecolor=edgecolor,
                        alpha=alpha,
                        label=f'class "{class_value}"',
                    )

                if i == j:
                    self.labels("value", "count", colname)
                else:
                    self.labels(colname, rowname, "")

                self.grid(True, alpha=0.3)
                self.legend()

    def labels(
        self,
        xlabel: str,
        ylabel: str,
        title: str,
        idx: Optional[int] = None,
        irow: Optional[int] = None,
        icol: Optional[int] = None,
    ) -> Any:
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


class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_dir.mkdir(exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _generate_cache_id(self, func_name: str, args: tuple, kwargs: dict) -> str:
        args_str = str(args) + str(sorted(kwargs.items()))
        
        hash_obj = hashlib.md5(args_str.encode())
        return f"{func_name}_{hash_obj.hexdigest()[:8]}"
    
    def _generate_checksum(self, args: tuple, kwargs: dict) -> str:
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(args_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_id: str, additional_name: Optional[str] = None) -> Path:
        if additional_name:
            filename = f"{additional_name}_{cache_id}.pkl"
        else:
            filename = f"{cache_id}.pkl"
        return self.cache_dir / filename
    
    def is_cache_valid(self, cache_path: str, current_checksum: str) -> bool:
        return (cache_path in self.metadata and 
                self.metadata[cache_path] == current_checksum)
    
    def update_cache_metadata(self, cache_path: str, checksum: str):
        self.metadata[str(cache_path)] = checksum
        self._save_metadata()
    
    def clear_cache(self):
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata = {}
        self._save_metadata()


cache_manager = CacheManager()


def cached(additional_name: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_id = cache_manager._generate_cache_id(func.__name__, args, kwargs)
            current_checksum = cache_manager._generate_checksum(args, kwargs)
            cache_path = cache_manager.get_cache_path(cache_id, additional_name)
            
            if cache_path.exists() and cache_manager.is_cache_valid(str(cache_path), current_checksum):
                try:
                    with open(cache_path, 'rb') as f:
                        print(f"\033[90m> Loading cached result for {func.__name__}\033[0m")
                        return pickle.load(f)
                except (pickle.PickleError, EOFError):
                    print("\033[90m> Cache file corrupted, recalculating...\033[0m")
            
            print(f"\033[90m> Calculating result for {func.__name__}...\033[0m")
            result = func(*args, **kwargs)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            cache_manager.update_cache_metadata(str(cache_path), current_checksum)
            
            return result
        
        return wrapper
    return decorator


def data_stats(data1, data2) -> Any:
    Pearson_v, Pearson_pv = scipy.stats.pearsonr(data1, data2)
    spearman_v, spearman_pv = scipy.stats.spearmanr(data1, data2)

    return np.array([Pearson_v, Pearson_pv, spearman_v, spearman_pv])


def get_cov_matrixes(df, features, classes):
    stats = np.zeros((len(features), len(features), (len(classes) + 1) * 4))

    for i, iparam in enumerate(features):
        for j, jparam in enumerate(features):
            data = np.array(df[[iparam, jparam, "target"]])

            stats[i, j, 0:4] = data_stats(data[..., 0], data[..., 1])

            for k, group in enumerate(classes):
                data_by_class = data[data[..., -1] == k]

                stats[i, j, (k + 1) * 4 : (k + 2) * 4] = data_stats(
                    data_by_class[..., 0], data_by_class[..., 1]
                )

    return stats


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



np.random.seed(42)
n_samples = 100
n_features = 8
n_classes = 3

centers = [
    [+1.0] * 8,
    [-1.0] * 8,
    [+1.0, -1.0] * 4,
]

cluster_std = np.random.uniform(0.2, 1.3, size=len(centers))

COLUMNS = [f"feature_{i + 1}" for i in range(n_features)]


@cached("base_df")
def create_dataframe(n_samples, n_features, n_classes, centers, cluster_std):
    X, y = make_blobs(
        n_samples=n_samples * n_classes,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=COLUMNS)
    df["target"] = y

    return df


df1 = create_dataframe(n_samples, n_features, n_classes, centers, cluster_std)

print(f"Размерность df1: {df1.shape}")
print(f"Количество классов: {len(df1['target'].unique())}")
print("Количество объектов в каждом классе:")
print(df1["target"].value_counts().sort_index())
print(df1.describe())

@cached()
def get_cov(df):
    cov_matrixes = get_cov_matrixes(df, COLUMNS, [0, 1, 2])

    return cov_matrixes


if CALC_COV:
    cov_matrixes = get_cov(df1)

    plotter = Plotter(nrows=4, ncols=4, figsize=(16, 16))

    class_names = ["all", "0", "1", "2"]
    for i in range(4):
        plotter.set_position(idx=i * 4)
        plotter.imshow(cov_matrixes[..., i * 4], cmap="winter", vmin=0.0, vmax=1.0)
        plotter.labels("", "", f"Pearson for {class_names[i]}")

        plotter.set_position(idx=i * 4 + 1)
        plotter.imshow(
            cov_matrixes[..., i * 4 + 1], cmap="winter", vmin=0.0, vmax=1.0
        )
        plotter.labels("", "", f"Pearson p-value for {class_names[i]}")

        plotter.set_position(idx=i * 4 + 2)
        plotter.imshow(
            cov_matrixes[..., i * 4 + 2], cmap="winter", vmin=0.0, vmax=1.0
        )
        plotter.labels("", "", f"Spearman for {class_names[i]}")

        plotter.set_position(idx=i * 4 + 3)
        im = plotter.imshow(
            cov_matrixes[..., i * 4 + 3], cmap="winter", vmin=0.0, vmax=1.0
        )
        plotter.labels("", "", f"Spearman p-value for {class_names[i]}")

        cbar = plotter.fig.colorbar(
            im, ax=plotter.position, orientation="vertical", pad=0.1
        )

    plotter.tight_layout()
    plotter.save("res/corelations.png")


    all_data = []
    groups = []
    classes = [0, 1, 2]

    for k in range(len(classes) + 1):
        group = classes[k - 1] if k > 0 else "all"
        groups.append(group)
        data = {
            "Pair": [],
            "Pearson": [],
            "Pearson p-value": [],
            "Spearman": [],
            "Spearman p-value": [],
        }
        local_matrix = cov_matrixes[..., (k * 4) : (k + 1) * 4]
        elems = ["Pearson", "Pearson p-value", "Spearman", "Spearman p-value"]
        for i, iparam in enumerate(COLUMNS):
            for j, jparam in enumerate(COLUMNS):
                data["Pair"].append(f"{iparam} x {jparam}")
                data["Pearson"].append(local_matrix[i, j, 0])
                data["Pearson p-value"].append(local_matrix[i, j, 1])
                data["Spearman"].append(local_matrix[i, j, 2])
                data["Spearman p-value"].append(local_matrix[i, j, 3])
        data = pd.DataFrame(data)
        data.to_csv(f"res/stats_for_{group}.csv", index=False)

        all_data.append(data)


    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(f"Stats for {groups[0]}")
    all_data[0]

    print(f"Stats for {groups[1]}")
    all_data[1]

    print(f"Stats for {groups[2]}")
    all_data[2]

    print(f"Stats for {groups[3]}")
    all_data[3]


X = df1[COLUMNS]
y = df1["target"]


plotter = Plotter(nrows=4, ncols=2, figsize=(10, 20))

for i, col in enumerate(COLUMNS):
    plotter.set_position(idx=i)

    for class_value in range(n_classes):
        mask = y == class_value
        plotter.hist(
            df1[mask][col],
            bins=30,
            color=COLORS[class_value],
            alpha=0.5,
            edgecolor=EDGECOLOR,
            label=f"class_{class_value}",
        )
    plotter.labels("value", "count", col)
    plotter.grid(True, alpha=0.3)
    plotter.legend()

plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/df1_hists.png", dpi=DPI)


if CALC_GRID:
    plotter = Plotter(nrows=8, ncols=8, figsize=(24, 24))

    plotter.dataset_visual(
        df1[COLUMNS], df1["target"], COLORS, STYLES, EDGECOLOR, alpha=0.6
    )

    plotter.tight_layout()
    plotter.save("res/df1_8x8.png", dpi=200)


@cached()
def make_AB(df):
    data = {"df": df}

    target_class = 0

    for a in [1, 10, 100]:
        for b in [1, 10, 100]:
            new_vals = {"target": [target_class] * a}
            t = -1
            for f in COLUMNS:
                new_vals[f] = [t * b] * a
                t *= -1

            new_obj = pd.DataFrame(new_vals)
            new_df = pd.concat([df, new_obj], ignore_index=True)
            
            data[f"df_{a}_{b}"] = new_df
    
    return data


def filter_data(df, classes, features):
    classes_remap = {classes[0]: 0, classes[1]: 1}
    df_filtered = df[df["target"].isin(classes)].copy()
    df_filtered["target"] = df_filtered["target"].map(classes_remap)
    X, y = df_filtered[features].values, df_filtered["target"].values

    return df_filtered, X, y


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def project_points(points, k, b):
    k_2 = -1 / k
    
    b_2 = points[..., 1] - k_2 * points[..., 0]

    x_proj = (b_2 - b) / (k - k_2)
    y_proj = k * x_proj + b
    
    return x_proj, y_proj


data = make_AB(df1)

feature = COLUMNS[2]
classes = [0, 1]

nrows = len(data.keys())
ncols = 3

plotter = Plotter(nrows=nrows, ncols=ncols, figsize=(12, 30))

for i, dataset in enumerate(data.keys()):
    ldf, lX, ly = filter_data(data[dataset], classes, [feature])
    
    X_plot = np.linspace(lX.min(), lX.max(), 100)

    lin_reg = LinearRegression()
    lin_reg.fit(lX, ly)
    
    beta = lin_reg.coef_[0]
    bias = lin_reg.intercept_
    
    points = np.hstack((lX, ly.reshape(-1, 1)))
    x_proj, y_proj = project_points(points, beta, bias)
    
    y_plot = lin_reg.predict(X_plot.reshape(-1, 1))
    
    plotter.set_position(idx=i * ncols)
    plotter.scatter(
        lX,
        ly,
        c=ldf["target"],
        cmap="RdYlGn",
        edgecolor=EDGECOLOR,
        alpha=0.6,
        label="Data"
    )
    plotter.scatter(
        x_proj,
        y_proj,
        color="blue",
        alpha=0.6,
    )
    plotter.plot(X_plot, y_plot, c="b", label="Linear Regression")
    plotter.labels(feature, "class", f"{dataset}")
    plotter.grid(True, alpha=0.3)
    plotter.legend()
    
    X_proj = lX.flatten()
    sorter = np.argsort(X_proj)
    X_proj = X_proj[sorter]
    y_proj = lin_reg.predict(X_proj.reshape(-1, 1))
    y_sigmoid = sigmoid(X_proj)
    
    plotter.set_position(idx=i * ncols + 1)
    plotter.scatter(
        lX,
        ly,
        c=ldf["target"],
        cmap="RdYlGn",
        edgecolor=EDGECOLOR,
        alpha=0.6,
        label="Data"
    )
    plotter.plot([lX.min(), lX.max()], [0.5] * 2, color="red", alpha=0.8, label="Standart threshold")
    plotter.plot(X_proj, y_sigmoid, c="b", label="Logistic Regression")
    plotter.labels("linreg", "class", f"{dataset}")
    plotter.grid(True, alpha=0.3)
    plotter.legend()

plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/linreg.png")
