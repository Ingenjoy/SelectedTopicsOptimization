# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 2016
Last update Wed May 2 2016

@author: michielfmstock@gmail.com

A relatively simple model to predict the antimicrobial activity of a peptide

Peptides are described by a set of features using (auto)correlation of the
physicochemical properties of the amino acids. These features are calculated
for a dataset of peptides showing antimicrobial activity. We subsequently use
a very simple kernel embedding model to represent the distribution of the
peptides.
"""

from sklearn.preprocessing import Normalizer
from protein_sequence_features import protein_features
import pandas as pd
import numpy as np
from Bio import SeqIO
from math import exp

print('Loading the sequences...')

peptides = []
for seq in SeqIO.parse('Data/anti_microbial_peptide.fasta', 'fasta'):
    peptides.append(seq)

print('Loading the features...')

if False:
    features_reference = np.vstack([protein_features(pep, lag_range=range(1, 5))
                for pep in peptides])
    pd.DataFrame(features_reference).to_csv('Data/anti_microbial_peptide_features.csv', index=False)

features_reference = pd.read_csv('Data/anti_microbial_peptide_features.csv')

normalizer = Normalizer()
features_reference = normalizer.fit_transform(features_reference.values)

def kernel_embedding_score(peptide_feature, gamma=0.01):
    """
    Calculates the score for kernel embedding
    """
    score = np.exp(-gamma * np.sum((peptide_feature - features_reference)**2,
                                                                    1)).mean()
    return score

def score_peptide(peptide, gamma=5):
    peptide_feature = protein_features(peptide,
                            lag_range=range(1, 5)).reshape((1, -1))
    peptide_feature[:] = normalizer.transform(peptide_feature)
    return kernel_embedding_score(peptide_feature, gamma)

print('Finished!')
