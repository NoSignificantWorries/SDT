import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.style.use("seaborn-v0_8-darkgrid")
np.random.seed(42)

n = 4
N = 1000
M = 1000
direction = np.array([1, 1, 1, 1], dtype=float)
direction = direction / np.linalg.norm(direction)  # нормируем, чтобы длина была 1

# Скалярные параметры вдоль прямой для "облака"
# Все точки будут лежать на одной прямой: t * direction
# t_cloud = np.random.randn(N) * 0.5  # можно сделать любое распределение по прямой
# cloud = np.outer(t_cloud, direction)  # форма (N, 4)

# Точка-выброс тоже на этой прямой, но далеко
# t_outlier = 5.0  # скаляр вдоль прямой
# outlier_point = t_outlier * direction  # форма (4,)
# heavy_outlier = np.tile(outlier_point, (M, 1))  # (M, 4)
cloud = np.random.randn(N, n) * 0.5
outlier_point = np.array([5] * n)
heavy_outlier = np.tile(outlier_point, (M, 1))

X_A = np.vstack([cloud, heavy_outlier])
labels_A = np.array([0] * N + [1] * M)
mask0 = labels_A == 0
mask1 = labels_A == 1

# Исходные данные (4x4 сетка)
fig, ax = plt.subplots(n, n, figsize=(n * 3, n * 3))
for i in range(n):
    for j in range(n):
        ax[i][j].scatter(
            X_A[mask0][..., i], X_A[mask0][..., j], color="green", alpha=0.4
        )
        ax[i][j].scatter(X_A[mask1][..., i], X_A[mask1][..., j], color="red", alpha=0.4)

plt.tight_layout()
plt.savefig("task4/4x4_1.png", dpi=200)

# ================= PCA =================
pca = PCA(n_components=2)
X_A2 = pca.fit_transform(X_A)
print(pca.explained_variance_ratio_)

print(X_A2[..., 0].mean(), X_A2[..., 0].min(), X_A2[..., 0].max())
print(X_A2[..., 1].mean(), X_A2[..., 1].min(), X_A2[..., 1].max())

# График точек в пространстве PC1–PC2
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_A2[mask0, 0], X_A2[mask0, 1], color="green", alpha=0.4, label="cloud")
ax.scatter(X_A2[mask1, 0], X_A2[mask1, 1], color="red", alpha=0.4, label="outlier")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.tight_layout()
plt.savefig("task4/pca.png", dpi=200)

# График нагрузок (векторы признаков + круг корреляций)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Для каждого признака вычисляем корреляцию с PC1 и PC2
loadings = []
for j in range(X_A.shape[1]):
    corr_pc1 = np.corrcoef(X_A[:, j], X_A2[:, 0])[0, 1]
    corr_pc2 = np.corrcoef(X_A[:, j], X_A2[:, 1])[0, 1]
    loadings.append((corr_pc1, corr_pc2))

loadings = np.array(loadings)

# Рисуем векторы (стрелки) для каждого признака
for i in range(loadings.shape[0]):
    x, y = loadings[i]
    ax.arrow(
        0,
        0,
        x,
        y,
        head_width=0.05,
        head_length=0.05,
        fc="blue",
        ec="blue",
        alpha=0.8,
        length_includes_head=True,
    )
    ax.text(x * 1.1, y * 1.1, f"Feature {i}", fontsize=10, ha="center", va="center")

# Рисуем единичную окружность (круг корреляций)
circle = Circle((0, 0), 1, color="black", fill=False, linestyle="--", alpha=0.5)
ax.add_patch(circle)

# Оформление графика
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal", "box")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True, alpha=0.3)
ax.set_title("PCA Loadings Plot (Correlation Circle)")

plt.tight_layout()
plt.savefig("task4/pca_loadings_circle.png", dpi=200)
plt.close()

# ================= t-SNE =================
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
X_tsne = tsne.fit_transform(X_A)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_tsne[mask0, 0], X_tsne[mask0, 1], color="green", alpha=0.4, label="cloud")
ax.scatter(X_tsne[mask1, 0], X_tsne[mask1, 1], color="red", alpha=0.4, label="outlier")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend()
ax.set_title("t-SNE projection (2D)")
plt.tight_layout()
plt.savefig("task4/tsne.png", dpi=200)
plt.close()

# ================= UMAP =================
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_A)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_umap[mask0, 0], X_umap[mask0, 1], color="green", alpha=0.4, label="cloud")
ax.scatter(X_umap[mask1, 0], X_umap[mask1, 1], color="red", alpha=0.4, label="outlier")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend()
ax.set_title("UMAP projection (2D)")
plt.tight_layout()
plt.savefig("task4/umap.png", dpi=200)
plt.close()

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from matplotlib.patches import Circle
#
# plt.style.use('seaborn-v0_8-darkgrid')
# np.random.seed(42)
#
# n = 4
# N = 1000
# M = 1000
# cloud = np.random.randn(N, n) * 0.5
# outlier_point = np.array([5] * n)
# heavy_outlier = np.tile(outlier_point, (M, 1))
#
# X_A = np.vstack([cloud, heavy_outlier])
# labels_A = np.array([0] * N + [1] * M)
# mask0 = labels_A == 0
# mask1 = labels_A == 1
#
# # Исходные данные (4x4 сетка)
# fig, ax = plt.subplots(n, n, figsize=(n * 3, n * 3))
# for i in range(n):
#         for j in range(n):
#                 ax[i][j].scatter(X_A[mask0][..., i], X_A[mask0][..., j], color="green", alpha=0.4)
#                 ax[i][j].scatter(X_A[mask1][..., i], X_A[mask1][..., j], color="red", alpha=0.4)
#
# plt.tight_layout()
# plt.savefig("task4/4x4_1.png", dpi=200)
#
# # PCA
# pca = PCA(n_components=2)
# X_A2 = pca.fit_transform(X_A)
#
# # График точек в пространстве PC1–PC2
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.scatter(X_A2[mask0, 0], X_A2[mask0, 1], color="green", alpha=0.4, label="cloud")
# ax.scatter(X_A2[mask1, 0], X_A2[mask1, 1], color="red", alpha=0.4, label="outlier")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.legend()
# plt.tight_layout()
# plt.savefig("task4/pca.png", dpi=200)
#
# # График нагрузок (векторы признаков + круг корреляций)
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#
# # Для каждого признака вычисляем корреляцию с PC1 и PC2
# loadings = []
# for j in range(X_A.shape[1]):
#         corr_pc1 = np.corrcoef(X_A[:, j], X_A2[:, 0])[0, 1]
#         corr_pc2 = np.corrcoef(X_A[:, j], X_A2[:, 1])[0, 1]
#         loadings.append((corr_pc1, corr_pc2))
#
# loadings = np.array(loadings)
#
# # Рисуем векторы (стрелки) для каждого признака
# for i in range(loadings.shape[0]):
#         x, y = loadings[i]
#         ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05,
#                               fc='blue', ec='blue', alpha=0.8, length_includes_head=True)
#         ax.text(x * 1.1, y * 1.1, f'Feature {i}', fontsize=10, ha='center', va='center')
#
#     # Рисуем единичную окружность (круг корреляций)
# circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5)
# ax.add_patch(circle)
#
# # Оформление графика
# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-1.2, 1.2)
# ax.set_aspect('equal', 'box')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.grid(True, alpha=0.3)
# ax.set_title('PCA Loadings Plot (Correlation Circle)')
#
# plt.tight_layout()
# plt.savefig("task4/pca_loadings_circle.png", dpi=200)
# plt.close()
#
