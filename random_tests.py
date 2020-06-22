import json
import os
from os import path
from pathlib import Path

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
# WAN and phi correlation

# \\\ Own libraries:
import Utils.dataTools

# \\\ Separate functions:
graphNormalizationType = 'rows'  # or 'cols' - Makes all rows add up to 1.
keepIsolatedNodes = False  # If True keeps isolated nodes
forceUndirected = True  # If True forces the graph to be undirected (symmetrizes)
forceConnected = True  # If True removes nodes (from lowest to highest degree)


dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

name = 'abbott'
data = Utils.dataTools.Authorship(name, 0.6, 0.2, dataPath)

# %%##################################################################
# WAN and phi correlation

nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]
nodesToKeep = []

G = graphTools.Graph('fuseEdges', nNodes,
                     data.selectedAuthor['train']['WAN'],
                     'sum', graphNormalizationType, keepIsolatedNodes,
                     forceUndirected, forceConnected, nodesToKeep)


phi_matrix_path='phi_matrices.txt'

with open(phi_matrix_path, 'r') as f:
    file = json.load(f)
    phi_whole = np.array(file[name]['phi'])[file['nodes']]

    corr = np.corrcoef(phi_whole, G.S)