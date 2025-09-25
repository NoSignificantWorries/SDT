import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sl
import scipy
import seaborn as sns
import statsmodels.api as sm

# Load Iris dataset
iris = sns.load_dataset("iris")


# Print Data Info
print(f"Dataframe size: {iris.shape[0]}x{iris.shape[1]}")
print(f"Count of features: {len(iris.columns) - 1}")
print(f"Num classes: {len(iris["species"].unique())}")
print("Class counts:")
for name, count in zip(*np.unique(iris["species"], return_counts=True)):
    print(f" - {name}: {count}")
null_values_count = iris.isnull().sum().sum()
print(f"Null values: {null_values_count} ({(iris.isnull().sum().sum() / iris.size * 100):.2f}%)")

if null_values_count > 0:
    iris = iris.dropna().reset_index(drop=True)


# Drawing dotted plots
count_of_features = 4
columns = iris.columns

iris["target"], unique_names = pd.factorize(iris["species"])

nrows = 4
ncols = 4
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 24))

styles = ["o", "s", "^"]
colors = ["red", "green", "blue"]

idx = 0
for i in range(count_of_features):
    for j in range(count_of_features):
        data_to_show = np.array(iris[[columns[i], columns[j], "target"]])
        lax = ax[idx // ncols][idx % ncols]
        for group in np.unique(data_to_show[..., -1]):
            local_data = data_to_show[data_to_show[..., -1] == group]
            lax.scatter(
                local_data[..., 0],
                local_data[..., 1],
                marker=styles[int(group)],
                color=colors[int(group)],
                alpha=0.7,
                label=unique_names[int(group)])
        lax.set_xlabel(columns[i])
        lax.set_ylabel(columns[j])
        lax.grid(True, alpha=0.3)
        ax[idx // ncols][idx % ncols].legend()
        idx += 1


# Drawing hists
nrows = 2
ncols = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))

for i in range(count_of_features):
    data_to_show = iris[columns[i]]
    ax[i // ncols][i % ncols].set_title(columns[i])
    ax[i // ncols][i % ncols].hist(data_to_show, bins=30, color="green", edgecolor="black", alpha=0.7)



def data_stats(data_table):
    pirson_matrix = np.zeros((count_of_features, count_of_features), dtype=np.float64)
    pirson_p_value = np.zeros((count_of_features, count_of_features), dtype=np.float64)
    spearman_matrix = np.zeros((count_of_features, count_of_features), dtype=np.float64)
    spearman_p_value = np.zeros((count_of_features, count_of_features), dtype=np.float64)

    for i in range(count_of_features):
        for j in range(count_of_features):
            pirson_v, pirson_pv = scipy.stats.pearsonr(data_table[columns[i]], data_table[columns[j]])
            pirson_matrix[i, j] = pirson_v
            pirson_p_value[i, j] = pirson_pv

            spearman_v, spearman_pv = scipy.stats.spearmanr(data_table[columns[i]], data_table[columns[j]])
            spearman_matrix[i, j] = spearman_v
            spearman_p_value[i, j] = spearman_pv


    drawing_query = [(pirson_matrix, "Pirson"),
                     (pirson_p_value, "Pirson p-value"),
                     (spearman_matrix, "Spearman"),
                     (spearman_p_value, "Spearman p-value")]
    
    print(drawing_query[0])
    print(drawing_query[1])
    print(drawing_query[2])
    print(drawing_query[3])
    
    return drawing_query


def draw_table(data_table):
    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))

    drawing_query = data_stats(data_table)

    img = []
    for i, (draw_data, draw_name) in enumerate(drawing_query):
        row, col = i // ncols, i % ncols

        img.append(ax[row][col].imshow(draw_data, cmap="coolwarm"))
        
        for i in range(draw_data.shape[0]):
            for j in range(draw_data.shape[1]):
                text = ax[row, col].text(j, i, np.round(draw_data[i, j], 4), ha="center", va="center", color="w")

        ax[row][col].set_title(draw_name)
        ax[row][col].set_xticks(np.arange(count_of_features))
        ax[row][col].set_yticks(np.arange(count_of_features))
        ax[row][col].set_xticklabels(columns[:count_of_features])
        ax[row][col].set_yticklabels(columns[:count_of_features])

    cbar = fig.colorbar(img[0], ax=ax)
    cbar.set_label("Values", rotation=270, labelpad=20)
            
draw_table(iris)
draw_table(iris[:][iris["target"] == 0])
draw_table(iris[:][iris["target"] == 1])
draw_table(iris[:][iris["target"] == 2])


# Drawing all the data
count_of_features = 4
columns = iris.columns

iris["target"], unique_names = pd.factorize(iris["species"])

nrows = 4
ncols = 4
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(48, 36))

styles = ["o", "s", "^"]
colors = ["red", "green", "blue"]


idx = 0
for i in range(count_of_features):
    for j in range(count_of_features):
        data_to_show = np.array(iris[[columns[i], columns[j], "target"]])
        lax = ax[idx // ncols][idx % ncols]
        for group in np.unique(data_to_show[..., -1]):
            local_data = data_to_show[data_to_show[..., -1] == group]
            
            if i != j:
                data_size = local_data.shape[0]
                
                x = local_data[..., 0].reshape((data_size, 1))
                y = local_data[..., 1]
                
                x_const = sm.add_constant(x)

                model = sm.OLS(y, x_const).fit()
                
                x_plot = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
                x_plot_with_const = sm.add_constant(x_plot)
                
                predictions = model.get_prediction(x_plot_with_const)
                frame = predictions.summary_frame(alpha=0.05)
                
                y_pred = frame['mean']
                ci_lower = frame['mean_ci_lower']
                ci_upper = frame['mean_ci_upper']
                pi_lower = frame['obs_ci_lower']
                pi_upper = frame['obs_ci_upper']
                
                lax.fill_between(x_plot.flatten(), pi_lower, pi_upper, 
                               color=colors[int(group)], alpha=0.2, 
                               label=f'95% Pred ({unique_names[int(group)]})')
                
                lax.fill_between(x_plot.flatten(), ci_lower, ci_upper, 
                               color=colors[int(group)], alpha=0.4,
                               label=f'95% Conf ({unique_names[int(group)]})')
                
                lax.plot(x_plot,
                         y_pred,
                         linewidth=4,
                         color=colors[int(group)],
                         label=f"LinReg ({unique_names[int(group)]})")

                lax.scatter(
                    local_data[..., 0],
                    local_data[..., 1],
                    marker=styles[int(group)],
                    color=colors[int(group)],
                    alpha=0.9,
                    label=unique_names[int(group)])
            else:
                lax.hist(local_data[..., 0],
                         bins=30, color=colors[int(group)],
                         alpha=0.5, edgecolor="black",
                         label=unique_names[int(group)])

        if i != j:
            lax.set_title(f"{columns[i]} x {columns[j]}")
            lax.set_xlabel(columns[i])
            lax.set_ylabel(columns[j])
        else:
            lax.set_title(columns[i])
            lax.set_xlabel("Values")
            lax.set_ylabel("Counts")
        lax.grid(True, alpha=0.3)
        ax[idx // ncols][idx % ncols].legend()
        idx += 1
