# coding: utf-8
from Bio import SeqIO
peptides = SeqIO('Data/anti_microbial_peptide.fasta', 'fasta')
peptides = SeqIO.parse('Data/anti_microbial_peptide.fasta', 'fasta')
peptides
peptides = []
for seq in SeqIO.p('Data/anti_microbial_peptide.fasta', 'fasta'):
    peptides.append(seq)
    
for seq in SeqIO.parse('Data/anti_microbial_peptide.fasta', 'fasta'):
    peptides.append(seq)
    
peptides
import pandas as pd
from protein_sequence_features import protein_features
from protein_sequence_features import protein_features
features = map(protein_features, peptides)
features[0]
peptides_dataset = pd.DataFrame(features, index=map(lambda x:x.id, peptides))
peptides_dataset
peptides_dataset.to_csv('Data/anti_microbial_peptide_features.csv')
from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit(peptides_dataset.values)
X = tsne.fit_transform(peptides_dataset.values)
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(X)
plt.scatter(X[:,0], X[:,1])
plt.imshow()
plt.show()
from sklearn.manifold import SpectralEmbedding
spectral_embedder = SpectralEmbedding()
X = spectral_embedder.fit_transform(peptides_dataset.values)
plt.scatter(X[:,0], X[:,1])
plt.show()
plt.scatter(X[:,0], X[:,1])
plt.xlabel('First component')
plt.ylabel('Second component')
plt.imsave('Figures/spectral_embedding_peptides.png')
plt.show()
from sklearn.svm import OneClassSVM
get_ipython().magic(u'pinfo OneClassSVM')
ocsvm = OneClassSVM()
ocsvm.fit(peptides_dataset.values)
peptides_dataset.shape
ocsvm.n_support_
ocsvm.C
ocsvm = OneClassSVM(C=1)
get_ipython().magic(u'pinfo OneClassSVM')
ocsvm.predict(peptides_dataset.values)
ocsvm.predict(peptides_dataset.values).std()
ocsvm.probability(peptides_dataset.values).std()
ocsvm.probability(peptides_dataset.values)
ocsvm.probability
get_ipython().magic(u'pinfo ocsvm.decision_function')
ocsvm.decision_function(peptides_dataset)
ocsvm.dual_coef_.shape
inlier = ocsvm.predict(peptides_dataset)
fig, ax = plt.subplots()
for i, lab_i in enumerate(inlier):
    if lab_i == 1:
        ax.scatter(X[i,0], X[i,0], c='b')
    else:
        ax.scatter(X[i,0], X[i,0], c='r')
        
ax.set_ylabel('Second component')
ax.set_xlabel('First component')
fig.show()
fig, ax = plt.subplots()
for i, lab_i in enumerate(inlier):
    if lab_i == 1:
        ax.scatter(X[i,0], X[i,1], c='b')
    else:
        ax.scatter(X[i,0], X[i,1], c='r')
        
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
fig.show()
X = tsne.fit_transform(peptides_dataset.values)
fig, ax = plt.subplots()
for i, lab_i in enumerate(inlier):
    if lab_i == 1:
        ax.scatter(X[i,0], X[i,1], c='b')
    else:
        ax.scatter(X[i,0], X[i,1], c='r')
        
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
fig.show()
from sklearn.preprocessing import normalize
X = tsne.fit_transform(normalize(peptides_dataset.values))
fig, ax = plt.subplots()
for i, lab_i in enumerate(inlier):
    if lab_i == 1:
        ax.scatter(X[i,0], X[i,1], c='b')
    else:
        ax.scatter(X[i,0], X[i,1], c='r')
        
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
fig.show()
ocsvm.coef0
ocsvm.coef_
ocsvm.dual_coef_
