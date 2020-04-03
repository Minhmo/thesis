import json
import os
from os import path

import matplotlib
import torch

import train_helper as th
from Utils import graphTools

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
RS = 123

# %%##################################################################
# Function for summarizing training results.
all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']


def summarize_training_results(train_result):
    author_best_acc = {}
    for author_name in train_result.keys():
        author_results = train_result[author_name]

        means = []
        stds = []

        for comb in author_results.keys():
            mean = np.mean(author_results[comb])
            std = np.std(author_results[comb])

            means.append(mean)
            stds.append(std)

        # best_comb = max(author_results.items(), key=operator.itemgetter(1))[0]
        # best_acc = max(author_results.items(), key=operator.itemgetter(1))[1]

        best_acc = max(means)
        index_of_best = means.index(best_acc)

        std = stds[index_of_best]
        best_comb = list(author_results.keys())[index_of_best]

        # best_combs.append(best_comb)
        author_best_acc[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    return author_best_acc


def analyse_GNN_results(train_result):
    means = []
    stds = []

    for comb in train_result.keys():
        mean = np.mean(train_result[comb])
        std = np.std(train_result[comb])

        means.append(mean)
        stds.append(std)

    best_acc = max(means)
    index_of_best = means.index(best_acc)
    std = stds[index_of_best]
    best_comb = list(train_result.keys())[index_of_best]

    return best_acc, best_comb, std


def analyse_GNN_results_extra(train_result):
    means_acc = []
    means_f1 = []
    means_auc = []
    stds = []

    for comb in train_result.keys():
        mean_acc = np.mean(train_result[comb]['acc'])
        mean_f1 = np.mean(train_result[comb]['f1'])
        mean_auc = np.mean(train_result[comb]['auc'])
        std = np.std(train_result[comb]['acc'])

        means_acc.append(mean_acc)
        means_f1.append(mean_f1)
        means_auc.append(mean_auc)
        stds.append(std)

    best_acc = max(means_acc)
    best_f1 = max(means_f1)
    best_auc = max(means_auc)

    index_of_best = means_acc.index(best_acc)
    std = stds[index_of_best]
    best_comb = list(train_result.keys())[index_of_best]

    return best_acc, best_comb, std, best_f1, best_auc


def prepare_for_training(data, order):
    data = torch.from_numpy(data).double()

    data_ordered = data[:, order].unsqueeze(1)

    return data_ordered


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def analyze_linear_results(json):
    results = {}

    for author in all_author_names:
        temp = json[author]

        if isinstance(temp, dict):
            means_knn = [np.mean(v) for k, v in temp.items()]
            max_acc = np.max(means_knn)
        else:
            max_acc = np.max(temp)

        results[author] = max_acc

    return results


# %%##################################################################
# compare the results of 1 and 2 layer feedforward networks

with open('1_feedforward_results_BCLoss.txt', 'r') as f:
    train_result = json.load(f)
    feedforward_results = summarize_training_results(train_result)

with open('2_feedforward_results_BCLoss.txt', 'r') as f:
    train_result = json.load(f)
    two_feedforward_results = summarize_training_results(train_result)

best_count = []

for author_name in feedforward_results.keys():
    if feedforward_results[author_name]['best_acc'] >= two_feedforward_results[author_name]['best_acc']:
        best_count.append(1)
    else:
        best_count.append(2)

# %%##################################################################
# group authors into bins
first = []
second = []
third = []

for author_name in two_feedforward_results.keys():
    acc = two_feedforward_results[author_name]['best_acc']

    if acc < 0.9:
        first.append(author_name)
    elif 0.9 <= acc < 0.95:
        second.append(author_name)
    elif 0.95 <= acc:
        third.append(author_name)

# %%##################################################################
# Anaylse linear model results. Compare to GCNN resutls.
all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

knn_file = open('results/knn_results.txt', 'r')
svm_file = open('results/svm_results.txt', 'r')

knn_search_results = json.load(knn_file)
svm_search_results = json.load(svm_file)

knn_results = analyze_linear_results(knn_search_results)
svm_results = analyze_linear_results(svm_search_results)

knn_file.close()
svm_file.close()

# %%##################################################################
# Anaylse GCNN results on random SO

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

BASE_FILE_NAME_RANDOM_SO = 'results/random_so/GNN_Polynomial_random_so_results_'
GCNN_random_so_results = dict.fromkeys(all_author_names, {'best_acc': None, 'std': None, 'best_comb': None})

for author_name in all_author_names:
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_RANDOM_SO, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_RANDOM_SO, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_random_so_results[author_name] = {'best_acc': np.mean(train_result[list(train_result.keys())[0]]),
                                               "best_comb": list(train_result.keys())[0],
                                               "std": np.std(train_result[list(train_result.keys())[0]])}

# %%##################################################################
# Anaylse GCNN results on classification by Nationality and Gender

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

FILE_NAME_NATIONALITY = 'GNN_Polynomial_nationality_results_20200310112333.txt'
FILE_NAME_GENDER = 'results/gender/GNN_Polynomial_gender_results_20200312161323.txt'

with open(FILE_NAME_NATIONALITY, 'r') as f:
    train_result = json.load(f)
    nationality_results = {'acc': np.mean(train_result['(64, 4)']['acc']),
                           'f1': np.mean(train_result['(64, 4)']['f1']),
                           'auc': np.mean(train_result['(64, 4)']['auc']),
                           "std": np.std(train_result['(64, 4)']['acc'])}

with open(FILE_NAME_GENDER, 'r') as f:
    train_result = json.load(f)

    means_acc = []
    means_f1 = []
    means_auc = []
    stds = []

    for comb in train_result.keys():
        mean_acc = np.mean(train_result[comb]['acc'])
        mean_f1 = np.mean(train_result[comb]['f1'])
        mean_auc = np.mean(train_result[comb]['auc'])
        std = np.std(train_result[comb]['acc'])

        means_acc.append(mean_acc)
        means_f1.append(mean_f1)
        means_auc.append(mean_auc)
        stds.append(std)

    best_acc = max(means_acc)
    best_f1 = max(means_f1)
    best_auc = max(means_auc)

    index_of_best = means_acc.index(best_acc)
    std = stds[index_of_best]
    best_comb = list(train_result.keys())[index_of_best]

    gender_results = {'best_acc': best_acc,
                      'best_f1': best_f1,
                      'best_auc': best_auc,
                      'best_comb': best_comb,
                      'std': std}

# %%##################################################################
# Anaylse results from using Phi matrix as a Shift operator

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

BASE_FILE_NAME_PHI = 'results/phi/GNN_Polynomial_phi_results_'
BASE_FILE_NAME_PHI_perc = 'results/phi_perc/GNN_Polynomial_phi_perc_results_'

GCNN_PHI_acc_results_all = {}
GCNN_PHI_perc_results_all = {}

GCNN_PHI_results = {}
GCNN_PHI_perc_results = {}

phi_acc = open('EdgeVariGNN_important_words_phi_accuracy3.txt', 'r')
phi_perc = open('EdgeVariGNN_important_words_phi.txt', 'r')

train_result_phi_acc = json.load(phi_acc)
train_result_phi_perc = json.load(phi_perc)

for author_name in all_author_names:
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_PHI, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_PHI, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_PHI_acc_results_all[author_name] = train_result
        best_acc, best_comb, std = analyse_GNN_results(GCNN_PHI_acc_results_all[author_name])
        GCNN_PHI_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    GCNN_PHI_results[author_name]['no_of_words'] = np.count_nonzero(train_result_phi_acc[author_name]['phi'])

    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_PHI_perc, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_PHI_perc, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_PHI_perc_results_all[author_name] = train_result
        best_acc, best_comb, std = analyse_GNN_results(GCNN_PHI_perc_results_all[author_name])
        GCNN_PHI_perc_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    GCNN_PHI_perc_results[author_name]['no_of_words'] = len(train_result_phi_perc[author_name]['indices'])

phi_acc.close()
phi_perc.close()
# %%##################################################################
# Compare GCNN results with FF results

with open('results/2_feedforward_results_BCLoss.txt', 'r') as f:
    train_result = json.load(f)
    two_feedforward_results = summarize_training_results(train_result)

BASE_FILE_NAME_1L = 'results/bc_loss_gnn/GNN_Polynomial_results_'
BASE_FILE_NAME_2L = 'results/Gnn_2_layers/GNN_Polynomial_2layers_results_'

GCNN_results = {}
GCNN_2layer_results = {}
GCNN_2layer_results_all = {}
GCNN_results_all = {}

for author_name in two_feedforward_results.keys():
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_1L, author_name)):
        continue

    with open('results/bc_loss_gnn/GNN_Polynomial_results_{0}.txt'.format(author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_results_all[author_name] = train_result

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_2L, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_2layer_results_all[author_name] = train_result

for author_name in GCNN_results_all.keys():
    best_acc, best_comb, std = analyse_GNN_results(GCNN_results_all[author_name])
    GCNN_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    best_acc, best_comb, std = analyse_GNN_results(GCNN_2layer_results_all[author_name])
    GCNN_2layer_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

best_count_acc = []
best_count_std = []

for author_name in GCNN_results.keys():
    # compare accuracies
    if GCNN_results[author_name]['best_acc'] >= two_feedforward_results[author_name]['best_acc']:
        best_count_acc.append(1)
    else:
        best_count_acc.append(2)

        # compare stds
    if GCNN_results[author_name]['std'] >= two_feedforward_results[author_name]['std']:
        best_count_std.append(1)
    else:
        best_count_std.append(2)

# create a dataframe with info from GCNN and FF2
model_comparison_df = pd.DataFrame.from_dict(GCNN_results)
model_comparison_df = model_comparison_df.T

model_comparison_df['best_acc_ff'] = [v['best_acc'] for k, v in two_feedforward_results.items()]
model_comparison_df['best_comb_ff'] = [v['best_comb'] for k, v in two_feedforward_results.items()]

# add linear model results
model_comparison_df['best_acc_svm'] = [v for k, v in svm_results.items()]
model_comparison_df['best_acc_knn'] = [v for k, v in knn_results.items()]

model_comparison_df['random_so'] = [v['best_acc'] for v in GCNN_random_so_results.values()]

# add 2 layer GCNN results
model_comparison_df['best_acc_2layer'] = [v['best_acc'] for k, v in GCNN_2layer_results.items()]
model_comparison_df['best_combination_2layer'] = [v['best_comb'] for k, v in GCNN_2layer_results.items()]

# add GCNN results using PHI as SO
model_comparison_df['best_acc_phi'] = [v['best_acc'] for k, v in GCNN_PHI_results.items()]
model_comparison_df['best_acc_perc_phi'] = [v['best_acc'] for k, v in GCNN_PHI_perc_results.items()]

model_comparison_df['non_zero_el_acc_phi'] = [v['no_of_words'] for k, v in GCNN_PHI_results.items()]
model_comparison_df['non_zero_el_perc_phi'] = [v['no_of_words'] for k, v in GCNN_PHI_perc_results.items()]

model_comparison_df = model_comparison_df[
    ['best_acc', 'random_so', 'best_acc_phi', 'best_acc_perc_phi', 'non_zero_el_acc_phi', 'non_zero_el_perc_phi',
     'best_acc_2layer',
     'best_acc_ff', 'best_acc_svm', 'best_acc_knn',
     'best_comb',
     'best_combination_2layer', 'best_comb_ff', 'std']]

# %%##################################################################
# Load data

# \\\ Own libraries:
import Utils.dataTools

# \\\ Separate functions:

dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

data = Utils.dataTools.Authorship('poe', 0.6, 0.2, dataPath)

# %%##################################################################
# run TSNE on the raw signals


X = np.concatenate((data.selectedAuthor['all']['wordFreq'], data.authorData['abbott']['wordFreq'],
                    ), axis=0)
# y = np.concatenate((np.array(['poe' for _ in range(data.selectedAuthor['all']['wordFreq'].shape[0])]),
#                     np.array(['abt' for _ in range(data.authorData['abbott']['wordFreq'].shape[0])])), axis=0)
y = np.concatenate((np.array([1 for _ in range(data.selectedAuthor['all']['wordFreq'].shape[0])]),
                    np.array([0 for _ in range(data.authorData['abbott']['wordFreq'].shape[0])]),
                    ), axis=0)

pca = PCA(n_components=4)
pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])
pca_df['pca1'] = pca_result[:, 0]
pca_df['pca2'] = pca_result[:, 1]
pca_df['pca3'] = pca_result[:, 2]
pca_df['pca4'] = pca_result[:, 3]

print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

top_two_comp = pca_df[['pca1', 'pca2']]
fashion_scatter(top_two_comp.values, y)

fashion_tsne = TSNE(random_state=RS, perplexity=20).fit_transform(X)
fashion_scatter(fashion_tsne, y)
plt.show()
plt.title("Raw signal")

# X_embedded = TSNE(n_components=2).fit_transform(X)
# X_embedded.shape

# %%##################################################################
# train model, with different parameters, use TSNE for dim renderExternalDocumentation

data.get_split('poe', 0.6, 0.2)

comb = ([1, 32], [2])

archit = th.train_net(data, comb)
net = archit.archit

###################################################################
#                                                                   #
#                    EMBEDDING HOOK                                 #
#                                                                   #
#####################################################################
import collections
from functools import partial

activations = collections.defaultdict(list)


def save_activation(name, mod, inp, out):
    activations[name].append(out.cpu())


# Registering hooks for all the Conv2d layers
# Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
# called repeatedly at different stages of the forward pass (like RELUs), this will save different
# activations. Editing the forward pass code to save activations is the way to go for these cases.
for name, m in net.named_modules():
    if name.strip() == "GFL.2":
        # partial to assign the layer name to each hook
        m.register_forward_hook(partial(save_activation, name))

X = np.concatenate((data.authorData['abbott']['wordFreq'],
                    data.authorData['poe']['wordFreq']), axis=0)

X = prepare_for_training(X, archit.order)

with torch.no_grad():
    net(X)

activations = activations['GFL.2'][0]
activations = activations.detach().numpy()
activations = activations.reshape((activations.shape[0], activations.shape[1] * activations.shape[2]))

y = np.concatenate((
    np.array([0 for _ in range(data.authorData['abbott']['wordFreq'].shape[0])]),
    np.array([1 for _ in range(data.authorData['poe']['wordFreq'].shape[0])])), axis=0)

fashion_tsne = TSNE(random_state=RS, perplexity=20).fit_transform(activations)
fashion_scatter(fashion_tsne, y)
plt.title(str(comb))
plt.show()

# %%##################################################################
# use this code to analysi PHI of edge variant GNNs

test = modelsGNN['EdgeVariGNN'].archit.EVGFL[0].Phi

phi = test[0, 0, 0, 0, :, :]
phi = phi.detach().numpy()

function_words = np.array(data.functionWords)
function_words = function_words[nodesToKeep]  # we get order from taining NN

important_pairs = [(function_words[x[0]] + " - " + function_words[x[1]]) for x in
                   np.argwhere(np.abs(phi) > np.max(phi) - 0.05 * np.max(phi))]

indices = [x for x in
           np.argwhere(np.abs(phi) > np.max(phi) - 0.05 * np.max(phi))]

result = {'indices': indices, 'nodes': nodesToKeep, 'order': order}

# %%##################################################################
# Plot scatter plot for GCNN combinations and dataset expolration results

plt.style.use('fivethirtyeight')

means = [it['meanDeg'] for it in author_info.values()]
F = [eval(it['best_comb'])[0][1] for it in GCNN_results.values()]

df_mean_degree = pd.DataFrame(columns=['mean degree', 'combination'], data={'mean degree': means, 'combination': F})
ax = sns.scatterplot(x="mean degree", y="combination", data=df_mean_degree)
plt.title("Mean degree vs number of Features")
plt.show()

df_best_acc_mean_degree = pd.DataFrame(columns=['Accuracy', 'Mean degree'],
                                       data={'Mean degree': means, 'Accuracy': model_comparison_df['best_acc']})
bx = sns.relplot(x="Mean degree", y="Accuracy", data=df_best_acc_mean_degree)
plt.title("Mean degree vs number of Accuracy")

plt.show()

# %%##################################################################
# Build subgraphs for each author
import networkx as nx
import pandas as pd

plt.style.use('fivethirtyeight')

function_words = np.array(data.functionWords)

with open('EdgeVariGNN_important_words_phi_accuracy3.txt', 'r') as f:
    training_results = json.load(f)

    for author_name in data.authorData.keys():
        current = training_results[author_name]
        current_words = function_words[current['nodes']]  # we get order from taining NN
        df = pd.DataFrame({'source': [current_words[x[0]] for x in current['indices']],
                           'target': [current_words[x[1]] for x in current['indices']]})

        G = nx.from_pandas_edgelist(df, 'source', 'target')
        fig = plt.figure()
        nx.draw(G, with_labels=True, alpha=0.6, node_size=2000, width=6.0, node_color="#008fd5", node_shape="o",
                font_size=25, linewidths=10)
        plt.show()
        fig.savefig("f_word_subgraph{0}.png".format(author_name))

# %%##################################################################
# Visualize subgraphs for each author

import networkx as nx
import pandas as pd
import igraph as i

print(i.__version__)
plt.style.use('fivethirtyeight')

function_words = np.array(data.functionWords)

with open('EdgeVariGNN_important_words_phi.txt', 'r') as f:
    training_results = json.load(f)

    for author_name in data.authorData.keys():
        current = training_results[author_name]
        current_words = function_words[current['nodes']]  # we get order from taining NN
        df = pd.DataFrame({'source': [current_words[x[0]] for x in current['indices']],
                           'target': [current_words[x[1]] for x in current['indices']],
                           'weight': [abs(current['phi'][x[0]][x[1]]) for x in current['indices']]})
        df.to_csv('results/important_words_subgraphs_percentage/' + author_name + '_subgraph.csv', index=False)

# %%##################################################################
# Build word map for each author.
edg_var_word_map = {}
edge_var_important_words = {}


def unique_author_words(name, all):
    other_words = []

    for k, v in all.items():
        if k == name:
            continue

        other_words.extend(all[k])

    return [x for x in all[name] if x not in other_words]


with open('EdgeVariGNN_important_words_phi.txt', 'r') as f:
    edge_var_important_words = json.load(f)
    function_words = np.array(data.functionWords)

    for author_name in data.authorData.keys():
        print(author_name)
        current = edge_var_important_words[author_name]

        indices = np.array(edge_var_important_words[author_name]['indices']).flatten()
        indices = np.unique(indices)

        current_function_words = function_words[current['nodes']]  # we get order from taining NN

        words = current_function_words[indices]

        edg_var_word_map[author_name] = words

# %%##################################################################
# plot correlation between unique words and Accuracy
from sklearn.preprocessing import normalize


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


unique_words = [len(unique_author_words(k, edg_var_word_map)) for k, v in edg_var_word_map.items()]
acc = model_comparison_df['best_acc'].tolist()

plt.style.use('fivethirtyeight')

df_unique_words_acc = pd.DataFrame(columns=['Accuracy', 'Unique words'],
                                   data={'Unique words': unique_words, 'Accuracy': acc})
bx = sns.relplot(x="Accuracy", y="Unique words", data=df_unique_words_acc)
plt.title("Unique words vs number of Accuracy")
plt.show()

# ########################################################
# try jaccard similarity

jaccard_distances = {}

for k, v in edg_var_word_map.items():
    distances = []

    for k2, v2 in edg_var_word_map.items():
        if k2 == k:
            continue

        distances.append(jaccard_similarity(v2, v))

    jaccard_distances[k] = np.mean(distances)

df_jaccard_acc = pd.DataFrame(columns=['Accuracy', 'Jaccard'],
                              data={'Jaccard': list(jaccard_distances.values()), 'Accuracy': acc})
bx = sns.lmplot(x="Accuracy", y="Jaccard", data=df_jaccard_acc)
plt.title("Jaccard vs number of Accuracy")
plt.show()

########################################################
# try degree and accuracy
acc = model_comparison_df['best_acc'].tolist()
df_degree_acc = pd.DataFrame(columns=['Accuracy', 'Degree'],
                             data={'Degree': [x.shape[0] for x in edg_var_word_map.values()], 'Accuracy': acc})
bx = sns.relplot(x="Accuracy", y="Degree", data=df_degree_acc)
plt.title("Degree vs Accuracy")
plt.show()

# %%##################################################################
# histogram of the most popular word pairs

from scipy.stats import norm
import collections

no_of_words_to_select = 20

all_selected_pairs = np.concatenate(list([v['pairs'] for v in edge_var_important_words.values()]))

pair_count = dict(collections.Counter(all_selected_pairs).most_common(no_of_words_to_select))
labels, values = zip(*pair_count.items())

indexes = np.arange(len(labels))
bar_width = 2.35

# Plot the histogram.
plt.bar(indexes, values, )
plt.xticks(indexes, labels, rotation='vertical')

values = list(pair_count.values())
mu, std = norm.fit(values)
plt.style.use('fivethirtyeight')
plt.show()

# df_pop_words_hist = pd.DataFrame(columns=['Word'],
#                                  data={'Word': all_selected_pairs})

# %%##################################################################
# histogram of the most popular words

from scipy.stats import norm
import collections

no_of_words_to_select = 50

all_selected_words = np.concatenate(list(edg_var_word_map.values()))

word_count = dict(collections.Counter(all_selected_words).most_common(no_of_words_to_select))
labels, values = zip(*word_count.items())

# not_used_words = set(function_words).difference(set(word_count.keys()))
#
# for w in not_used_words:
#     word_count[w] = 0

indexes = np.arange(len(labels))
bar_width = 2.35

# Plot the histogram.
plt.bar(indexes, values)
plt.xticks(indexes, labels, rotation='vertical')

# values = list(word_count.values())
# mu, std = norm.fit(values)
plt.style.use('fivethirtyeight')
#
# xnew = np.linspace(0, no_of_words_to_select, 300)  # 300 represents number of points to make between T.min and T.max
# power_smooth = spline(indexes, values, xnew)
# plt.plot(xnew, power_smooth)

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "The most popular words"
# plt.title(title)

plt.show()

df_pop_words_hist = pd.DataFrame(columns=['Word'],
                                 data={'Word': all_selected_words})

# %%##################################################################
# Correlation between signal and PHI

graphNormalizationType = 'rows'  # or 'cols' - Makes all rows add up to 1.
keepIsolatedNodes = False  # If True keeps isolated nodes
forceUndirected = True  # If True forces the graph to be undirected (symmetrizes)
forceConnected = True  # If True removes nodes (from lowest to highest degree)

correlation = []
for author_name in edge_var_important_words.keys():
    phi = np.array(edge_var_important_words[author_name]['phi'])
    # phi = phi.flatten()
    data.get_split(author_name, 0.6, 0.2)
    nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]

    signal = graphTools.Graph('fuseEdges', nNodes,
                              data.selectedAuthor['train']['WAN'],
                              'sum', graphNormalizationType, keepIsolatedNodes,
                              forceUndirected, forceConnected, [])

    corr = np.corrcoef(phi, signal.S[edge_var_important_words[author_name]['nodes']])
    correlation.append(corr)

# %%##################################################################
# Correlation between no of non zero elements in phi and accuracy

plt.style.use('fivethirtyeight')

df_acc_non_zero_phi_perc = pd.DataFrame()
df_acc_non_zero_phi_perc['Non zero elements'] = model_comparison_df['non_zero_el_perc_phi'].tolist()
df_acc_non_zero_phi_perc['Accuracy'] = model_comparison_df['best_acc_perc_phi'].tolist()

bx = sns.relplot(x="Accuracy", y="Non zero elements", data=df_acc_non_zero_phi_perc)
plt.show()

# %%##################################################################
# unique vs popular words for learning on GCNN

BASE_FILE_NAME_POPULAR = 'results/popular words GCNN/GNN_Polynomial_popular_words_results_'
BASE_FILE_NAME_UNIQUE = 'results/unique words GCNN/GNN_Polynomial_unique_words_results_'

GCNN_PUPULAR_results = {}
GCNN_UNIQUE_results = {}

for author_name in all_author_names:
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_POPULAR, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_POPULAR, author_name), 'r') as f:
        train_result = json.load(f)
        best_acc, best_comb, std = analyse_GNN_results(train_result)
        GCNN_PUPULAR_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_UNIQUE, author_name), 'r') as f:
        train_result = json.load(f)
        best_acc, best_comb, std = analyse_GNN_results(train_result)
        GCNN_UNIQUE_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

df_unique_popular = pd.DataFrame(GCNN_PUPULAR_results)
df_unique_popular = df_unique_popular.T

df_unique_popular['Unique'] = [v['best_acc'] for k, v in GCNN_UNIQUE_results.items()]

# %%##################################################################
# Correlation between 2nd eigenvalue and accuracy

with open('author_info.json', 'r') as f:
    author_info = json.load(f)

    # Best acc and Fiedler eigenvalue
    df_best_acc_fiedler_eigenvalue = pd.DataFrame(columns=['Accuracy', 'Algebraic connectivity'],
                                                  data={'Algebraic connectivity': [x['eigenvalues_avg'][1] for x in
                                                                                   author_info.values()],
                                                        'Accuracy': model_comparison_df['best_acc'].tolist()})
    bx = sns.lmplot(x="Algebraic connectivity", y="Accuracy", data=df_best_acc_fiedler_eigenvalue)
    plt.show()

    df_best_acc_fiedler_eigenvalue['Filter taps'] = [eval(it['best_comb'])[1][0] for it in GCNN_results.values()]
    bx = sns.lmplot(x="Algebraic connectivity", y="Filter taps", data=df_best_acc_fiedler_eigenvalue)
    plt.show()

    df_best_acc_fiedler_eigenvalue['Diameter'] = [x['diameter'] for x in author_info.values()]
    bx = sns.relplot(x="Diameter", y="Filter taps", data=df_best_acc_fiedler_eigenvalue)
    plt.show()

    bx = sns.relplot(x="Diameter", y="Accuracy", data=df_best_acc_fiedler_eigenvalue)
    plt.show()

# %%##################################################################
# correlation between accuracy and non-zero els of matrix

with open('GNN_Polynomial_phi_non_zero_results_poe.txt', 'r') as f:
    phi_non_zero = json.load(f)

    means = []
    stds = []
    non_zero_count = []

    for k, v in phi_non_zero.items():
        means.append(np.mean(v['acc']))
        stds.append(np.std(v['acc']))
        non_zero_count.append(v['non_zero'])

    df_acc_non_zero = pd.DataFrame()

    df_acc_non_zero['acc'] = means
    df_acc_non_zero['stds'] = stds
    df_acc_non_zero['non zero'] = non_zero_count

    bx = sns.relplot(x="acc", y="non zero", data=df_acc_non_zero)
    plt.show()

# %%##################################################################
# Gender classification result exploration

GENDER_EDGE = 'results/gender/EdgeVariGNN_Gender_results_20200316165853.txt'
GENDER_GCNN_2L = 'results/gender/2_layer_GNN_Polynomial_gender_results_20200316125137.txt'
GENDER_GCNN = 'results/gender/GNN_Polynomial_gender_results_20200312161323.txt'
GENDER_LINEAR_KNN = 'results/gender/knn_results_gender.txt'
GENDER_LINEAR_SVM = 'results/gender/svm_results_gender.txt'
df_gender_comparison = pd.DataFrame(index=['best_acc', 'best_comb', 'std', 'best_f1', 'best_auc'])

with open(GENDER_EDGE, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['Edge_net'] = list(result.values())

with open(GENDER_GCNN_2L, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['GCNN_2_layers'] = list(result.values())

with open(GENDER_GCNN, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['GCNN'] = list(result.values())

with open(GENDER_LINEAR_KNN, 'r') as f:
    train_result = json.load(f)
    result = analyze_linear_results(train_result)
    result = {'best_acc': result['abbott'], 'best_comb': '', 'std': '', 'best_f1': '', 'best_auc': ''}
    df_gender_comparison['GENDER_LINEAR_KNN'] = list(result.values())

with open(GENDER_LINEAR_SVM, 'r') as f:
    train_result = json.load(f)
    result = analyze_linear_results(train_result)
    result = {'best_acc': result['abbott'], 'best_comb': '', 'std': '', 'best_f1': '', 'best_auc': ''}
    df_gender_comparison['GENDER_LINEAR_SVM'] = list(result.values())

# create a dataframe with info from GCNN and FF2
# df_gender_comparison = df_gender_comparison.T
