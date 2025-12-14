# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **Task4**
#

# %% [markdown]
# ### **Stage 0:** _Importing libs. Adding plotter class and sys functions_
#

# %%
import json
import hashlib
from pathlib import Path

# import sys
import pickle
from typing import Tuple, Optional, Any, Callable
from functools import wraps

# data
import numpy as np
import pandas as pd

# visualization
# import matplotlib

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes

# data workflows
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import pacmap
from sklearn.manifold import TSNE


# %% [markdown]
# ##### **_Constants_**
#

# %%
DPI = 150
SAVE_DIR = "res"
COLORS = ["green", "red", "blue"]
COLORS_FOR_KGL = [
    "#f38ba8",
    "#eba0ac",
    "#fab387",
    "#f9e2af",
    "#a6e3a1",
    "#94e2d5",
    "#89dceb",
    "#74c7ec",
    "#89b4fa",
    "#b4befe",
]
EDGECOLOR = "#11111b"
STYLES = ["o", "s", "^"]
STYLES_FOR_KGL = ["o"] * 10

CACHED = False
RECACHE = False
CLEAR_CACHE = False

SHOW_CORRELATIONS = True
SHOW_HISTS = True
SHOW_GRID = True
SHOW_METHODS = True
SHOW_PCA_LDA = True
SHOW_LDA = True
SHOW_MNIST = True


# %% [markdown]
# ##### **_Plotter interface_**
#

# %%
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

    def scree_plot(self, pca, title: str):
        explained_var = pca.explained_variance_ratio_

        explained_variance = pca.explained_variance_

        n_components_kaiser = np.sum(explained_variance >= 1)

        def broken_stick_variances(n_components):
            variances = []
            for i in range(1, n_components + 1):
                variance = (
                    sum(1.0 / k for k in range(i, n_components + 1)) / n_components
                )
                variances.append(variance)
            return np.array(variances)

        n_total = len(explained_variance)
        expected_variances = broken_stick_variances(n_total)

        explained_variance_ratio = pca.explained_variance_ratio_
        n_components_broken_stick = np.sum(
            explained_variance_ratio > expected_variances
        )

        x = range(1, len(explained_var) + 1)
        self.plot(x, explained_var, "r-", marker="o", label="Scree plot")

        for (
            x_coord,
            y_coord,
        ) in zip(x, explained_var):
            self.annotate(
                f"{y_coord:.4f}",
                xy=(x_coord, y_coord),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )

        self.plot(
            x,
            expected_variances,
            "b--",
            label=f"Broken stick (n = {n_components_broken_stick})",
        )
        total_variance = sum(explained_variance)
        kaiser_line = 1 / total_variance
        self.axhline(
            y=kaiser_line,
            color="g",
            linestyle="--",
            label=f"Kaiser line (n = {n_components_kaiser})",
        )

        self.set_ylabel("Eigenvalue")
        self.set_xlabel("Eigenvalue number")
        self.set_ylim([0, 1.1])

        self.set_title(f"Scree Plot of {title}")
        self.set_xticks(range(1, len(explained_var) + 1))
        self.grid(True, alpha=0.3)
        self.legend()

    def components_projection(
        self, pca, pca_df, title, pca_columns, columns, colors=COLORS, styles=STYLES
    ):
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        for i, row_pca in enumerate(pca_columns):
            for j, col_pca in enumerate(pca_columns):
                self.set_position(irow=i, icol=j)

                for cls in pca_df["target"].unique():
                    cls_mask = pca_df["target"] == cls
                    self.scatter(
                        pca_df[cls_mask][row_pca],
                        pca_df[cls_mask][col_pca],
                        color=colors[cls],
                        marker=styles[cls],
                        alpha=0.5,
                        s=20,
                        label=f"Class {cls}",
                    )
                ax = self._get_axes_by_coords(irow=i, icol=j)

                x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
                y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

                scale_factor = min(x_range, y_range) / 10

                for var_idx in range(loadings.shape[0]):
                    self.arrow(
                        0,
                        0,
                        loadings[var_idx, i],
                        loadings[var_idx, j],
                        fc="k",
                        ec="k",
                        alpha=0.5,
                        width=0.04 * scale_factor,
                        head_width=0.3 * scale_factor,
                        head_length=0.5 * scale_factor,
                    )
                    self.text(
                        loadings[var_idx, i] * 1.15,
                        loadings[var_idx, j] * 1.15,
                        f"{columns[var_idx]}",
                        color="k",
                        ha="center",
                        va="center",
                    )

                self.set_xlabel(
                    f"{row_pca} (Var: {pca.explained_variance_ratio_[i]:.2%})"
                )
                self.set_ylabel(
                    f"{col_pca} (Var: {pca.explained_variance_ratio_[j]:.2%})"
                )
                self.legend()
                self.grid(True, alpha=0.3)
                self.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                self.axvline(x=0, color="k", linestyle="--", alpha=0.3)
                self.set_title(
                    f"{row_pca} vs {col_pca} with Variable Vectors for {title}"
                )

    def old_and_new_correlation(self, pca, title, pca_columns, columns):
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        for i, row_pca in enumerate(pca_columns):
            for j, col_pca in enumerate(pca_columns):
                self.set_position(irow=i, icol=j)

                circle = plt.Circle((0, 0), 1, fill=False, color="blue", alpha=0.3)
                self.add_patch(circle)

                for var_idx in range(loadings.shape[0]):
                    self.arrow(
                        0,
                        0,
                        loadings[var_idx, i],
                        loadings[var_idx, j],
                        fc="k",
                        ec="k",
                        alpha=0.7,
                    )

                    self.text(
                        loadings[var_idx, i] * 1.15,
                        loadings[var_idx, j] * 1.15,
                        f"{columns[var_idx]}" if columns else f"V{var_idx + 1}",
                        color="b",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

                    self.plot(
                        [0, loadings[var_idx, i]],
                        [0, loadings[var_idx, j]],
                        "k--",
                        alpha=0.2,
                    )

                self.set_xlim([-1.2, 1.2])
                self.set_ylim([-1.2, 1.2])

                self.axhline(y=0, color="k", linestyle="-", alpha=0.3)
                self.axvline(x=0, color="k", linestyle="-", alpha=0.3)

                self.set_xlabel(
                    f"{row_pca} (Var: {pca.explained_variance_ratio_[i]:.2%})"
                )
                self.set_ylabel(
                    f"{col_pca} (Var: {pca.explained_variance_ratio_[j]:.2%})"
                )
                self.set_title(
                    f"Correlation Circle: {row_pca} vs {col_pca} for {title}"
                )
                self.grid(True, alpha=0.3)

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



# %% [markdown]
# ##### **_Cache manager_**
#

# %%
class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_dir.mkdir(exist_ok=True)
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _generate_cache_id(self, func_name: str, args: tuple, kwargs: dict) -> str:
        args_str = str(args) + str(sorted(kwargs.items()))

        hash_obj = hashlib.md5(args_str.encode())
        return f"{func_name}_{hash_obj.hexdigest()[:8]}"

    def _generate_checksum(self, args: tuple, kwargs: dict) -> str:
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(args_str.encode()).hexdigest()

    def get_cache_path(
        self, cache_id: str, additional_name: Optional[str] = None
    ) -> Path:
        if additional_name:
            filename = f"{additional_name}_{cache_id}.pkl"
        else:
            filename = f"{cache_id}.pkl"
        return self.cache_dir / filename

    def is_cache_valid(self, cache_path: str, current_checksum: str) -> bool:
        return (
            cache_path in self.metadata
            and self.metadata[cache_path] == current_checksum
        )

    def update_cache_metadata(self, cache_path: str, checksum: str):
        self.metadata[str(cache_path)] = checksum
        self._save_metadata()

    def clear_cache(self):
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata = {}
        self._save_metadata()



# %%
cache_manager = CacheManager()
if CLEAR_CACHE or RECACHE:
    cache_manager.clear_cache()


def cached(additional_name: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_id = cache_manager._generate_cache_id(func.__name__, args, kwargs)
            current_checksum = cache_manager._generate_checksum(args, kwargs)
            cache_path = cache_manager.get_cache_path(cache_id, additional_name)

            if CACHED and not RECACHE:
                if cache_path.exists() and cache_manager.is_cache_valid(
                    str(cache_path), current_checksum
                ):
                    try:
                        with open(cache_path, "rb") as f:
                            print(
                                f"\033[90m> Loading cached result for {func.__name__}\033[0m"
                            )
                            return pickle.load(f)
                    except (pickle.PickleError, EOFError):
                        print("\033[90m> Cache file corrupted, recalculating...\033[0m")

                print(f"\033[90m> Calculating result for {func.__name__}...\033[0m")
                result = func(*args, **kwargs)

                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                cache_manager.update_cache_metadata(str(cache_path), current_checksum)
            elif RECACHE and CACHED:
                print(f"\033[90m> Calculating result for {func.__name__}...\033[0m")
                result = func(*args, **kwargs)

                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                cache_manager.update_cache_metadata(str(cache_path), current_checksum)
            else:
                print(f"\033[90m> Calculating result for {func.__name__}...\033[0m")
                result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator



# %% [markdown]
# ##### **_Data statistics_**
#

# %%
@cached()
def data_stats(data1, data2) -> Any:
    Pearson_v, Pearson_pv = scipy.stats.pearsonr(data1, data2)
    spearman_v, spearman_pv = scipy.stats.spearmanr(data1, data2)

    return np.array([Pearson_v, Pearson_pv, spearman_v, spearman_pv])


@cached()
def get_cov_matrixes(df, features):
    stats = np.zeros((len(features), len(features), 4))

    for i, iparam in enumerate(features):
        for j, jparam in enumerate(features):
            data = np.array(df[[iparam, jparam]])

            stats[i, j, 0:4] = data_stats(data[..., 0], data[..., 1])

    return stats



# %% [markdown]
# ### **Stage 1** _Generating data for first tasks_
#

# %%
np.random.seed(42)
COLUMNS = [f"axis_{i + 1}" for i in range(5)]


@cached()
def generate_dataset(A, k, n=1000):
    line_points = np.linspace(0, A, n)
    base_data = np.vstack([line_points] * 5).T
    variances = np.array([A / k, A / k, A / (k * 2), A / (k * 4), A / (k * 8)])
    noise = np.random.randn(n, 5) * np.sqrt(variances)
    data = base_data + noise
    df = pd.DataFrame(data, columns=COLUMNS)
    df["target"] = 0
    return df


A = 20
df = generate_dataset(A, 4)

print(f"Размерность df: {df.shape}")
print(df.describe())
print(df.head())


# %%
if SHOW_CORRELATIONS:
    cov_matrixes = get_cov_matrixes(df, COLUMNS)

    plotter = Plotter(nrows=2, ncols=2, figsize=(16, 16))

    for i, label in enumerate(
        ["Pearson", "Pearson p-value", "Spearman", "Spearman p-value"]
    ):
        plotter.set_position(idx=i)
        plotter.imshow(cov_matrixes[..., i], cmap="inferno", vmin=0.0, vmax=1.0)
        plotter.labels("", "", label)

    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/corelations.png")


# %%
all_data = []
groups = []

data = {
    "Pair": [],
    "Pearson": [],
    "Pearson p-value": [],
    "Spearman": [],
    "Spearman p-value": [],
}
elems = ["Pearson", "Pearson p-value", "Spearman", "Spearman p-value"]
for i, iparam in enumerate(COLUMNS):
    for j, jparam in enumerate(COLUMNS):
        data["Pair"].append(f"{iparam} x {jparam}")
        data["Pearson"].append(cov_matrixes[i, j, 0])
        data["Pearson p-value"].append(cov_matrixes[i, j, 1])
        data["Spearman"].append(cov_matrixes[i, j, 2])
        data["Spearman p-value"].append(cov_matrixes[i, j, 3])
data = pd.DataFrame(data)
data.to_csv(f"{SAVE_DIR}/stats.csv", index=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

data


# %% [markdown]
# ##### **_Hists_**
#

# %%
plotter = Plotter(nrows=5, ncols=1, figsize=(10, 20))

for i, col in enumerate(COLUMNS):
    plotter.set_position(idx=i)

    plotter.hist(
        df[col],
        bins=30,
        color="green",
        alpha=0.5,
        edgecolor=EDGECOLOR,
        label="Features hist",
    )
    plotter.labels("value", "count", col)
    plotter.grid(True, alpha=0.3)
    plotter.legend()

plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/df_hists.png", dpi=DPI)

# %% [markdown]
# ##### **_Show grid_**
#

# %%
if SHOW_GRID:
    plotter = Plotter(nrows=5, ncols=5, figsize=(20, 20))

    plotter.dataset_visual(
        df[COLUMNS], df["target"], COLORS, STYLES, EDGECOLOR, alpha=0.6
    )

    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/df_5x5.png", dpi=200)


# %%
@cached()
def calc_out(base_df):
    df_out = []
    m = 1000
    for i in [1, 2, 5, 10]:
        row = (A * i, -A * i, A * i, -A * i, A * i, 0)
        local_df = base_df.copy()

        new_df = pd.DataFrame([row] * m, columns=df.columns)
        local_df = pd.concat([local_df, new_df], ignore_index=True)

        df_out.append(local_df)

    return df_out


df_out = calc_out(df)

plotter = Plotter(nrows=5, ncols=5, figsize=(20, 20))

plotter.dataset_visual(
    df_out[-1][COLUMNS], df_out[-1]["target"], COLORS, STYLES, EDGECOLOR, alpha=0.6
)

plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/df_out_5x5.png", dpi=200)


# %% [markdown]
# ### **Stage 2:** _Working with PCA_
#

# %%
def make_pca_plots(
    ldf, name, pca, scaler, columns, components, colors=COLORS, styles=STYLES
):
    data_scaled = scaler.fit_transform(ldf[columns])
    pca_df = pca.fit_transform(data_scaled)
    PCA_COLUMNS = [f"PC{i + 1}" for i in range(components)]
    pca_df = pd.DataFrame(pca_df, columns=PCA_COLUMNS)
    pca_df["target"] = ldf["target"]

    plotter = Plotter(1, 1, figsize=(10, 10))
    plotter.set_position(idx=0)
    plotter.scree_plot(pca, name)
    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/scree_{name}.png")

    sz = components * 4
    plotter = Plotter(components, components, figsize=(sz, sz))
    plotter.dataset_visual(
        pca_df[PCA_COLUMNS], pca_df["target"], colors, styles, EDGECOLOR
    )
    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/pca_without_vectors_for_{name}.png", dpi=200)

    plotter = Plotter(components, components, figsize=(sz, sz))
    plotter.components_projection(
        pca, pca_df, name, PCA_COLUMNS, columns, colors=colors, styles=styles
    )
    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/pca_with_vectors_for_{name}.png", dpi=200)

    plotter = Plotter(components, components, figsize=(sz, sz))
    plotter.old_and_new_correlation(pca, name, PCA_COLUMNS, columns)
    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/pca_with_old_obj_for_{name}.png", dpi=200)

    return pca_df



# %%
pca = PCA()
scaler = StandardScaler()

pca_df = make_pca_plots(df, "df", pca, scaler, COLUMNS, 5)

# %%
pca_out_1_df = make_pca_plots(df_out[0], "df_out_1", pca, scaler, COLUMNS, 5)

# %%
pca_out_2_df = make_pca_plots(df_out[1], "df_out_2", pca, scaler, COLUMNS, 5)

# %%
pca_out_5_df = make_pca_plots(df_out[2], "df_out_5", pca, scaler, COLUMNS, 5)

# %%
pca_out_10_df = make_pca_plots(df_out[3], "df_out_10", pca, scaler, COLUMNS, 5)


# %% [markdown]
# ### **Stage 3:** _Trying other methods (UMAP, PacMAP, t-SNE)_
#

# %%
def make_methods(n_components):
    return {
        "UMAP": {
            "func": lambda seed, pca_df: umap.UMAP(
                n_components=n_components,
                random_state=seed,
                n_neighbors=min(15, pca_df.shape[0] // 10),
                min_dist=0.1,
                metric="euclidean",
            ).fit_transform(pca_df)
        },
        "PaCMAP": {
            "func": lambda seed, pca_df: pacmap.PaCMAP(
                n_components=n_components,
                random_state=int(seed),
                n_neighbors=None,
                MN_ratio=0.5,
                FP_ratio=2.0,
            ).fit_transform(pca_df)
        },
        "t-SNE": {
            "func": lambda seed, pca_df: TSNE(
                n_components=n_components,
                random_state=seed,
                perplexity=min(30, pca_df.shape[0] // 10),
                max_iter=1000,
                init="random",
            ).fit_transform(pca_df)
        },
    }


# %%
PROJ_COLUMNS = ["C1", "C2"]


@cached()
def calc_projections(seeds, methods, df, pca_df, columns):
    PCA_COLUMNS = pca_df.columns[:-1]
    res = {}
    for name in methods.keys():
        res[name] = {}
        for j, seed in enumerate(seeds):
            projections = methods[name]["func"](seed, df[columns])
            proj_df = pd.DataFrame(projections, columns=PROJ_COLUMNS)
            proj_df["target"] = df["target"]
            res[name][f"with seed {j}"] = proj_df
        projections_pca = methods[name]["func"](seed, pca_df[PCA_COLUMNS])
        proj_df = pd.DataFrame(projections_pca, columns=PROJ_COLUMNS)
        proj_df["target"] = pca_df["target"]
        res[name]["after PCA"] = proj_df

    return res


def show_projections(proj, df_name, colors=COLORS, styles=STYLES):
    count_methods = len(proj.keys())
    count_plots = len(proj[list(proj.keys())[0]].keys())
    plotter = Plotter(count_methods, count_plots, figsize=(22, 15))

    for i, name in enumerate(proj.keys()):
        for j, add_title in enumerate(proj[name].keys()):
            plotter.set_position(irow=i, icol=j)

            lproj = proj[name][add_title]
            for cls in lproj["target"].unique():
                plotter.scatter(
                    lproj[PROJ_COLUMNS[0]],
                    lproj[PROJ_COLUMNS[1]],
                    color=colors[cls],
                    marker=styles[cls],
                    alpha=0.4,
                    label=f"Class {cls}",
                )
            plotter.grid(True, alpha=0.3)
            plotter.labels("C1", "C2", f"{name} for {df_name} {add_title}")
            plotter.legend()

    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/UMAP_PacMAP_t-SNE_for_{df_name}.png", dpi=DPI)



# %%
n_random_starts = 4


@cached()
def get_random_seeds(n_random):
    return np.random.randint(0, 10000, n_random)


random_seeds = get_random_seeds(n_random_starts)

methods = make_methods(2)


# %%
if SHOW_METHODS:
    proj_df = calc_projections(random_seeds, methods, df, pca_df, COLUMNS)
    show_projections(proj_df, "df")

# %%
if SHOW_METHODS:
    proj_df_out_1 = calc_projections(
        random_seeds, methods, df_out[0], pca_out_1_df, COLUMNS
    )
    show_projections(proj_df_out_1, "df_out_1")

# %%
if SHOW_METHODS:
    proj_df_out_2 = calc_projections(
        random_seeds, methods, df_out[1], pca_out_2_df, COLUMNS
    )
    show_projections(proj_df_out_2, "df_out_2")

# %%
if SHOW_METHODS:
    proj_df_out_5 = calc_projections(
        random_seeds, methods, df_out[2], pca_out_5_df, COLUMNS
    )
    show_projections(proj_df_out_5, "df_out_5")

# %%
if SHOW_METHODS:
    proj_df_out_10 = calc_projections(
        random_seeds, methods, df_out[3], pca_out_10_df, COLUMNS
    )
    show_projections(proj_df_out_10, "df_out_10")


# %% [markdown]
# ### **Stage 0:** _Working with LDA_
#

# %%
@cached()
def generate_dataset_2(A, n, columns):
    start1 = np.zeros(4)
    end1 = np.array([A, A, A, A])

    start2 = np.array([A / 100, 0, 0, 0])
    end2 = np.array([A + A / 100, A, A, A])

    var = A / 200000
    sigma = np.sqrt(var)

    rng = np.random.default_rng(0)

    t1 = rng.random(n)
    t2 = rng.random(n)

    X1 = start1 + np.outer(t1, end1 - start1) + rng.normal(0, sigma, size=(n, 4))
    X2 = start2 + np.outer(t2, end2 - start2) + rng.normal(0, sigma, size=(n, 4))

    df_LDA = pd.DataFrame(np.vstack([X1, X2]), columns=columns)
    df_LDA["target"] = np.array([0] * n + [1] * n)

    return df_LDA


COLUMNS2 = [f"axes_{i}" for i in range(4)]
df_LDA = generate_dataset_2(20, 1000, COLUMNS2)

plotter = Plotter(4, 4, figsize=(16, 16))
plotter.dataset_visual(df_LDA[COLUMNS2], df_LDA["target"], COLORS, STYLES, EDGECOLOR)
plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/df_LDA.png", dpi=DPI)


# %%
pca = PCA()
scaler = StandardScaler()

if SHOW_PCA_LDA:
    pca_LDA_df = make_pca_plots(df_LDA, "df_LDA", pca, scaler, COLUMNS2, 4)


# %%
if SHOW_LDA:
    plotter = Plotter(4, 4, figsize=(16, 16))

    X = df_LDA[COLUMNS2]
    y = df_LDA["target"]
    size = 500

    for i, row_name in enumerate(COLUMNS2):
        for j, col_name in enumerate(COLUMNS2):
            loc_X = np.array(X[[row_name, col_name]])
            model = LinearDiscriminantAnalysis()
            model.fit(loc_X, y)

            x_min, x_max = loc_X[:, 0].min() - 1, loc_X[:, 0].max() + 1
            y_min, y_max = loc_X[:, 1].min() - 1, loc_X[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, size), np.linspace(y_min, y_max, size)
            )

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plotter.set_position(irow=i, icol=j)

            plotter.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
            plotter.scatter(
                loc_X[:, 0],
                loc_X[:, 1],
                c=y,
                cmap="bwr",
                alpha=0.4,
                edgecolors=EDGECOLOR,
                s=30,
                label="Objects",
            )
            plotter.legend()
            plotter.labels(row_name, col_name, f"LDA for {row_name} and {col_name}")

    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/LDA.png", dpi=DPI)


# %% [markdown]
# ### **Stage 4:** _Downloading and working with kaggle dataset_
#

# %%
# path = kagglehub.dataset_download("muratkokludataset/dry-bean-dataset")
# print("Path to dataset files:", path)

# with open('data/Dry_Bean_Dataset.arff', 'r', encoding='utf-8') as f:
#     data, meta = arff.loadarff(f)
#
# df = pd.DataFrame(data)

kgl_df = pd.read_csv("data/dry_bean.csv")
# kgl_df.to_csv("data/dry_bean.csv")
# kgl_df.to_csv("data/dry_bean.csv", index=False)
kgl_df = kgl_df.rename(columns={"Class": "target"})

# path = kagglehub.dataset_download("zalando-research/fashionmnist")
# kgl_df = pd.read_csv("data/mnist.csv")
# kgl_df = kgl_df.rename(columns={'label': 'target'})
# kgl_df.to_csv("data/mnist.csv")

KGL_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

unique_classes = kgl_df["target"].unique()
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
kgl_df["target"] = kgl_df["target"].map(class_to_idx)

for i in range(4):
    plotter = Plotter(4, 4, (16, 16))
    plotter.dataset_visual(
        kgl_df[KGL_COLUMNS[i * 4 : i * 4 + 4]],
        kgl_df["target"],
        COLORS_FOR_KGL,
        STYLES_FOR_KGL,
        EDGECOLOR,
        alpha=0.6,
    )
    plotter.tight_layout()
    plotter.save(f"{SAVE_DIR}/kgl_{i}.png", dpi=DPI)


# %%
components = 8
pca = PCA(n_components=components)
scaler = StandardScaler()

pca_kgl_df = make_pca_plots(
    kgl_df,
    "kgl_df",
    pca,
    scaler,
    KGL_COLUMNS,
    components,
    colors=COLORS_FOR_KGL,
    styles=STYLES_FOR_KGL,
)


# %%
components = 3
pca = PCA(n_components=components)
scaler = StandardScaler()

pca_kgl_filtered_df = make_pca_plots(
    kgl_df,
    "kgl_filtered_df",
    pca,
    scaler,
    KGL_COLUMNS,
    components,
    colors=COLORS_FOR_KGL,
    styles=STYLES_FOR_KGL,
)


# %%
n_random_starts = 2
random_seeds = get_random_seeds(n_random_starts)
methods = make_methods(2)

proj_kgl_df = calc_projections(
    random_seeds, methods, kgl_df, pca_kgl_filtered_df, KGL_COLUMNS
)
show_projections(proj_kgl_df, "kgl_df", colors=COLORS_FOR_KGL, styles=STYLES_FOR_KGL)

