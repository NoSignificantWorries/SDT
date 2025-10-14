# %%
import os
import sys
from typing import Tuple, Optional, Any
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
from scipy import stats
import time
from sklearn.utils import resample

# %%
DPI = 150
SAVE_DIR="res"
COLORS = ["#e78284", "#a6d189", "#b4befe"]
EDGECOLOR = "#11111b"
STYLES = ["o", "s", "^"]


class Plotter:
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: Tuple[int, int] = (6, 6)) -> None:
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.nrows = 0
        self.ncols = 0
        
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


def cache_dataframe(cache_file):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_file):
                print(f"> Loading from cache: {cache_file}")
                return pd.read_pickle(cache_file)
            else:
                print("> Generating a new dataset...")
                df = func(*args, **kwargs)
                df.to_pickle(cache_file)
                print(f"> Data loaded to cache: {cache_file}")
                return df
        return wrapper
    return decorator

# %%
def generate_new_dataset(n_samples, n_features, n_classes):
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

# =============================================================================
# ЗАДАНИЕ 2.1 - Генерация датасета df1
# =============================================================================
print("=== ЗАДАНИЕ 2.1 - Генерация датасета df1 ===")

np.random.seed(42)
n_samples = 1000
n_features = 16
n_classes = 3

COLUMNS = [f"feature_{i+1}" for i in range(n_features)]


@cache_dataframe("cache/df1_main.pkl")
def create_dataframe(n_samples, n_features, n_classes):
    return generate_new_dataset(n_samples, n_features, n_classes)

df1 = create_dataframe(n_samples, n_features, n_classes)

print(f"Размерность df1: {df1.shape}")
print(f"Количество классов: {len(df1["target"].unique())}")
print("Количество объектов в каждом классе:")
print(df1["target"].value_counts().sort_index())

# %%
X = df1[COLUMNS]
y = df1["target"]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

plotter = Plotter(nrows=1, ncols=1, figsize=(10, 8))
plotter.set_position(idx=0)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

for class_value in range(n_classes):
    mask = y == class_value
    plotter.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.5, marker=STYLES[class_value], color=COLORS[class_value], label=f"class_{class_value}")

plotter.labels("PCA axes 1", "PCA axes 2", "PCA visualization")
plotter.grid(True, alpha=0.3)
plotter.legend()
plotter.tight_layout()
plotter.save(f"{SAVE_DIR}/data_PCA.png", dpi=DPI)

# %%
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

# %%
# =============================================================================
# ЗАДАНИЕ 2.2 - Создание датасетов с повторенными объектами
# =============================================================================
print("\n=== ЗАДАНИЕ 2.2 - Создание датасетов с повторенными объектами ===")

# Класс 1 будем повторять (класс с пересечением)
class_to_repeat = 1
repetition_factors = [2, 5, 10, 20, 50, 100, 1000, 10000]

datasets = {"df1": df1}

for factor in repetition_factors:
    df_name = f"df{factor}"
    
    @cache_dataframe(f"cache/{df_name}.pkl")
    def generate_copy(base_df):
        new_df = base_df.copy()
        class_samples = df1[df1["target"] == class_to_repeat]
        repeated_samples = pd.concat([class_samples] * (factor - 1), ignore_index=True)
        new_df = pd.concat([new_df, repeated_samples], ignore_index=True)
        
        return new_df
    
    new_df = generate_copy(df1)
    datasets[df_name] = new_df
    print(f"Создан {df_name}: {new_df.shape} объектов")

DATASET_NAMES = list(datasets.keys())

sys.exit(0)

# %%
# =============================================================================
# ЗАДАНИЕ 2.3 - Визуализация LDA для разных датасетов
# =============================================================================
print("\n=== ЗАДАНИЕ 2.3 - Визуализация LDA для разных датасетов ===")

# Выбираем пару классов и признаки для визуализации
selected_classes = [0, 1]  # Класс 1 - тот, который повторяем
selected_features = ["feature_1", "feature_2"]

# Создаем фигуру для визуализации
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

# Словарь для хранения времени выполнения
lda_times = {}

for i, (df_name, ax) in enumerate(zip(DATASET_NAMES, axes)):
    if df_name not in datasets:
        continue
        
    print(df_name)

    df = datasets[df_name]
    
    # Фильтруем только выбранные классы
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    # Подготавливаем данные
    X_plot = df_filtered[selected_features].values
    y_plot = df_filtered["target"].values
    
    # Время начала обучения
    start_time = time.time()
    
    # Обучаем LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_plot, y_plot)
    
    # Время окончания обучения
    end_time = time.time()
    lda_times[df_name] = end_time - start_time
    
    # Создаем сетку для построения решающей границы
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Предсказываем для всех точек сетки
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Визуализация
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, 
                        cmap=plt.cm.RdYlBu, edgecolors="black", s=30)
    
    ax.set_title(f"{df_name}\nВремя: {lda_times[df_name]:.4f}с", fontsize=12)
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])

print("Время выполнения LDA для разных датасетов:")
for df_name, t in lda_times.items():
    print(f"{df_name}: {t:.4f} секунд")

plt.tight_layout()
plt.savefig("lda_decision_boundaries.png", dpi=200, bbox_inches="tight")


sys.exit(0)
# %%
# =============================================================================
# ЗАДАНИЕ 2.4 - Центры масс классов
# =============================================================================
print("\n=== ЗАДАНИЕ 2.4 - Центры масс классов ===")

# Создаем таблицу для центров масс
centers_data = []

for df_name in ["df1", "df2", "df5", "df10", "df20", "df50", "df100", "df1000"]:
    if df_name not in datasets:
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    centers = {}
    for class_label in selected_classes:
        class_data = df_filtered[df_filtered["target"] == class_label][selected_features]
        center = class_data.mean().values
        centers[class_label] = center
    
    # Общий центр масс
    overall_center = df_filtered[selected_features].mean().values
    
    # Центр отрезка между центрами классов
    midpoint = (centers[selected_classes[0]] + centers[selected_classes[1]]) / 2
    
    centers_data.append({
        "dataset": df_name,
        "center_class_0": centers[selected_classes[0]],
        "center_class_1": centers[selected_classes[1]],
        "overall_center": overall_center,
        "midpoint": midpoint
    })

# Создаем DataFrame с центрами масс
centers_df = pd.DataFrame(centers_data)
print("Центры масс для разных датасетов:")
print(centers_df)

# Визуализация центров масс для df1
df_name = "df1"
df = datasets[df_name]
df_filtered = df[df["target"].isin(selected_classes)].copy()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, df_name in enumerate(["df1", "df2", "df5", "df10", "df20", "df50", "df100", "df1000"]):
    if df_name not in datasets or i >= len(axes):
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    X_plot = df_filtered[selected_features].values
    y_plot = df_filtered["target"].values
    
    # Обучаем LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_plot, y_plot)
    
    # Создаем сетку
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Визуализация
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    axes[i].scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, 
                   cmap=plt.cm.RdYlBu, edgecolors="black", s=20, alpha=0.7)
    
    # Отмечаем центры масс
    centers_info = centers_df[centers_df["dataset"] == df_name].iloc[0]
    
    axes[i].scatter(centers_info["center_class_0"][0], centers_info["center_class_0"][1], 
                   c="red", marker="X", s=200, label="Center Class 0", edgecolors="black")
    axes[i].scatter(centers_info["center_class_1"][0], centers_info["center_class_1"][1], 
                   c="blue", marker="X", s=200, label="Center Class 1", edgecolors="black")
    axes[i].scatter(centers_info["overall_center"][0], centers_info["overall_center"][1], 
                   c="green", marker="*", s=200, label="Overall Center", edgecolors="black")
    axes[i].scatter(centers_info["midpoint"][0], centers_info["midpoint"][1], 
                   c="purple", marker="D", s=150, label="Midpoint", edgecolors="black")
    
    # Соединяем центры масс линией
    axes[i].plot([centers_info["center_class_0"][0], centers_info["center_class_1"][0]],
                [centers_info["center_class_0"][1], centers_info["center_class_1"][1]],
                "k--", alpha=0.7)
    
    axes[i].set_title(f"{df_name}")
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("lda_centers_visualization.png", dpi=300, bbox_inches="tight")
#plt.show()

# =============================================================================
# ЗАДАНИЕ 2.5 - ROC кривые с доверительными полосами
# =============================================================================
print("\n=== ЗАДАНИЕ 2.5 - ROC кривые с доверительными полосами ===")

def bootstrap_roc_curve(X, y, classifier, n_bootstrap=1000, target_class=1):
    """Бутстреп для построения доверительных полос ROC кривой"""
    tprs = []
    base_fpr = np.linspace(0, 1, 100)
    auc_scores = []
    
    for i in range(n_bootstrap):
        # Бутстреп выборка
        X_resampled, y_resampled = resample(X, y, random_state=i)
        
        # Обучаем классификатор
        classifier.fit(X_resampled, y_resampled)
        
        # Получаем вероятности для целевого класса
        if hasattr(classifier, "predict_proba"):
            y_score = classifier.predict_proba(X_resampled)[:, target_class]
        else:
            y_score = classifier.decision_function(X_resampled)
        
        # Вычисляем ROC кривую
        fpr, tpr, _ = roc_curve(y_resampled, y_score, pos_label=target_class)
        
        # Интерполируем до базового FPR
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        
        # Вычисляем AUC
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
    
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    
    return base_fpr, mean_tpr, tprs_lower, tprs_upper, auc_scores

# Визуализация ROC кривых для разных датасетов
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

roc_auc_data = {}

for i, df_name in enumerate(DATASET_NAMES):
    if df_name not in datasets or i >= len(axes):
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    X_data = df_filtered[selected_features].values
    y_data = df_filtered["target"].values
    
    # Бинаризуем метки для ROC анализа (целевой класс = 1)
    y_binary = (y_data == class_to_repeat).astype(int)
    
    # Бутстреп ROC кривых
    lda = LinearDiscriminantAnalysis()
    fpr, mean_tpr, tpr_lower, tpr_upper, auc_scores = bootstrap_roc_curve(
        X_data, y_binary, lda, n_bootstrap=100, target_class=1)
    
    mean_auc = np.mean(auc_scores)
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    
    roc_auc_data[df_name] = {
        "mean_auc": mean_auc,
        "auc_ci": auc_ci,
        "auc_scores": auc_scores
    }
    
    # Визуализация ROC кривой
    axes[i].plot(fpr, mean_tpr, color="b", label=f"ROC (AUC = {mean_auc:.3f})")
    axes[i].fill_between(fpr, tpr_lower, tpr_upper, color="grey", alpha=0.3,
                        label="95% доверительный интервал")
    axes[i].plot([0, 1], [0, 1], "k--", alpha=0.5)
    
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel("False Positive Rate")
    axes[i].set_ylabel("True Positive Rate")
    axes[i].set_title(f"ROC кривая - {df_name}\nAUC: {mean_auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    axes[i].legend(loc="lower right")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_curves_with_ci.png", dpi=300, bbox_inches="tight")
#plt.show()

print("Метрики AUROC для разных датасетов:")
for df_name, metrics in roc_auc_data.items():
    print(f"{df_name}: AUC = {metrics["mean_auc"]:.3f} [{metrics["auc_ci"][0]:.3f}, {metrics["auc_ci"][1]:.3f}]")

# =============================================================================
# ЗАДАНИЕ 2.6 - PR кривые
# =============================================================================
print("\n=== ЗАДАНИЕ 2.6 - PR кривые ===")

def bootstrap_pr_curve(X, y, classifier, n_bootstrap=1000, target_class=1):
    """Бутстреп для построения доверительных полос PR кривой"""
    precisions = []
    base_recall = np.linspace(0, 1, 100)
    auc_scores = []
    
    for i in range(n_bootstrap):
        X_resampled, y_resampled = resample(X, y, random_state=i)
        classifier.fit(X_resampled, y_resampled)
        
        if hasattr(classifier, "predict_proba"):
            y_score = classifier.predict_proba(X_resampled)[:, target_class]
        else:
            y_score = classifier.decision_function(X_resampled)
        
        precision, recall, _ = precision_recall_curve(y_resampled, y_score, pos_label=target_class)
        
        # Интерполируем precision для базового recall
        precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
        precisions.append(precision_interp)
        
        # Вычисляем AUPRC
        auprc = auc(recall, precision)
        auc_scores.append(auprc)
    
    precisions = np.array(precisions)
    mean_precision = precisions.mean(axis=0)
    std_precision = precisions.std(axis=0)
    
    precision_upper = np.minimum(mean_precision + 1.96 * std_precision, 1)
    precision_lower = np.maximum(mean_precision - 1.96 * std_precision, 0)
    
    return base_recall, mean_precision, precision_lower, precision_upper, auc_scores

# Визуализация PR кривых
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

pr_auc_data = {}

for i, df_name in enumerate(DATASET_NAMES):
    if df_name not in datasets or i >= len(axes):
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    X_data = df_filtered[selected_features].values
    y_data = df_filtered["target"].values
    y_binary = (y_data == class_to_repeat).astype(int)
    
    # Бутстреп PR кривых
    lda = LinearDiscriminantAnalysis()
    recall, mean_precision, precision_lower, precision_upper, auc_scores = bootstrap_pr_curve(
        X_data, y_binary, lda, n_bootstrap=100, target_class=1)
    
    mean_auprc = np.mean(auc_scores)
    auprc_ci = np.percentile(auc_scores, [2.5, 97.5])
    
    pr_auc_data[df_name] = {
        "mean_auprc": mean_auprc,
        "auprc_ci": auprc_ci
    }
    
    # Визуализация PR кривой
    axes[i].plot(recall, mean_precision, color="b", label=f"PR (AUPRC = {mean_auprc:.3f})")
    axes[i].fill_between(recall, precision_lower, precision_upper, color="grey", alpha=0.3,
                        label="95% доверительный интервал")
    
    # Базовая линия - доля положительных классов
    baseline = np.mean(y_binary)
    axes[i].axhline(y=baseline, color="r", linestyle="--", 
                   label=f"Baseline = {baseline:.3f}")
    
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel("Recall")
    axes[i].set_ylabel("Precision")
    axes[i].set_title(f"PR кривая - {df_name}\nAUPRC: {mean_auprc:.3f} [{auprc_ci[0]:.3f}, {auprc_ci[1]:.3f}]")
    axes[i].legend(loc="upper right")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pr_curves_with_ci.png", dpi=300, bbox_inches="tight")
#plt.show()

print("Метрики AUPRC для разных датасетов:")
for df_name, metrics in pr_auc_data.items():
    print(f"{df_name}: AUPRC = {metrics["mean_auprc"]:.3f} [{metrics["auprc_ci"][0]:.3f}, {metrics["auprc_ci"][1]:.3f}]")

# =============================================================================
# ЗАДАНИЕ 2.7 - ROC кривые для другого целевого класса
# =============================================================================
print("\n=== ЗАДАНИЕ 2.7 - ROC кривые для другого целевого класса ===")

# Выбираем другой целевой класс
alternative_target_class = 0

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for i, df_name in enumerate(DATASET_NAMES):
    if df_name not in datasets or i >= len(axes):
        continue
        
    df = datasets[df_name]
    df_filtered = df[df["target"].isin(selected_classes)].copy()
    
    X_data = df_filtered[selected_features].values
    y_data = df_filtered["target"].values
    
    # Бинаризуем для альтернативного целевого класса
    y_binary = (y_data == alternative_target_class).astype(int)
    
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
    axes[i].set_title(f"ROC (Класс {alternative_target_class}) - {df_name}\nAUC: {mean_auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    axes[i].legend(loc="lower right")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_curves_alternative_class.png", dpi=300, bbox_inches="tight")
#plt.show()

# =============================================================================
# ЗАДАНИЕ 2.8 - Кросс-валидация для df10
# =============================================================================
print("\n=== ЗАДАНИЕ 2.8 - Кросс-валидация для df10 ===")

df10 = datasets["df10"]
df_filtered = df10[df10["target"].isin(selected_classes)].copy()

X_data = df_filtered[selected_features].values
y_data = df_filtered["target"].values
y_binary = (y_data == class_to_repeat).astype(int)

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

# =============================================================================
# ЗАДАНИЕ 2.9 - LDA с разным количеством признаков
# =============================================================================
print("\n=== ЗАДАНИЕ 2.9 - LDA с разным количеством признаков ===")

feature_counts = [2, 4, 8, 16]
df1_data = datasets["df1"]
df_filtered = df1_data[df1_data["target"].isin(selected_classes)].copy()

y_binary = (df_filtered["target"] == class_to_repeat).astype(int)

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
    y_binary = (y_data == class_to_repeat).astype(int)
    
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

