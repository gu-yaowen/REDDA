"""
Title:Negative DDA Sampling Based on Biological Network Path
Descripition:
For each drug, a biological network path search is executed to find out the disease that the drug
has a low probability (at leaset 5 edges for a drug-disease link) to associated to, which we called
neg-DDA sampling.
Input Args:the Csv file of bilogical network, the List of candidate drugs, the List of candidate
diseases, number of negative sampling to execute for each drug.
Output Shape: a DataFrame including positive data (label 1) and negative data (label 0): ['Drug',
 'Disease', 'Label']
"""

import numpy as np
import pandas as pd
import networkx as nx


# 所有ID已改变，需要重新洗数据
#########################
def neg_sampling(g, n_neg, drugs, diseases):
    spl = nx.all_pairs_shortest_path_length(g)
    neg_drug, neg_disease = [], []
    count = 0
    for node_id, node_matrix in spl:
        if node_id in drugs:
            drug_disease = dict([(key, node_matrix[key]) for key in node_matrix.keys() if key in diseases])
            disease_id = list(drug_disease.keys())
            neg_dis_idx = np.where(np.array(list(drug_disease.values())) >= 5)[0]
            neg_dis_idx = np.random.choice(neg_dis_idx, n_neg, replace=False).tolist()
            neg_dis = np.array(disease_id)[neg_dis_idx]
            neg_drug.extend([node_id for i in range(n_neg)])
            neg_disease.extend([dis for dis in neg_dis])
            count += 1
            if count % 100 == 0:
                print('Sampling {}/{}'.format(count, len(drugs)))
        elif set(neg_drug) == set(drugs):
            print('Sampling Finished!')
            break
    return pd.DataFrame(np.array([neg_drug, neg_disease]).T, columns=['Drug', 'Disease'])


np.random.seed(0)

drug = pd.read_csv('.\\omics\\drug.csv')
protein = pd.read_csv('.\\omics\\protein.csv')
gene = pd.read_csv('.\\omics\\gene.csv')
pathway = pd.read_csv('.\\omics\\pathway.csv')
disease = pd.read_csv('.\\omics\\disease.csv')
drug_drug = pd.read_csv('.\\interactions\\drug_drug.csv').values
drug_drug = [tuple(drug_drug[i, :].tolist()) for i in range(len(drug_drug))]
protein_protein = pd.read_csv('.\\interactions\\protein_protein.csv').values
protein_protein = [tuple(protein_protein[i, :].tolist()) for i in range(len(protein_protein))]
gene_gene = pd.read_csv('.\\interactions\\gene_gene.csv').values
gene_gene = [tuple(gene_gene[i, :-1].tolist()) for i in range(len(gene_gene))]
pathway_pathway = pd.read_csv('.\\interactions\\pathway_pathway.csv').values
pathway_pathway = [tuple(pathway_pathway[i, :].tolist()) for i in range(len(pathway_pathway))]
disease_disease = pd.read_csv('.\\interactions\\disease_disease.csv').values
disease_disease = [tuple(disease_disease[i, :].tolist()) for i in range(len(disease_disease))]
drug_protein = pd.read_csv('.\\associations\\drug_protein.csv').values
drug_protein = [tuple(drug_protein[i, :].tolist()) for i in range(len(drug_protein))]
drug_disease = pd.read_csv('.\\associations\\drug_disease.csv').values
drug_disease = [tuple(drug_disease[i, :].tolist()) for i in range(len(drug_disease))]
protein_gene = pd.read_csv('.\\associations\\protein_gene.csv').values
protein_gene = [tuple(protein_gene[i, :].tolist()) for i in range(len(protein_gene))]
gene_pathway = pd.read_csv('.\\associations\\gene_pathway.csv').values
gene_pathway = [tuple(gene_pathway[i, :].tolist()) for i in range(len(gene_pathway))]
pathway_disease = pd.read_csv('.\\associations\\pathway_disease.csv').values
pathway_disease = [tuple(pathway_disease[i, :].tolist()) for i in range(len(pathway_disease))]

g = nx.Graph()
g.clear()
g.add_nodes_from(drug['Drug'].values)
g.add_nodes_from(protein['Protein'].values)
g.add_nodes_from(gene['Gene'].values)
g.add_nodes_from(pathway['Pathway'].values)
g.add_nodes_from(disease['Disease'].values)
# g.add_weighted_edges_from(drug_drug)
g.add_edges_from(protein_protein)
g.add_edges_from(gene_gene)
g.add_edges_from(pathway_pathway)
# g.add_weighted_edges_from(disease_disease)
g.add_edges_from(drug_protein)
g.add_edges_from(drug_disease)
g.add_edges_from(protein_gene)
g.add_edges_from(gene_pathway)
g.add_edges_from(pathway_disease)

drug_disease = pd.read_csv('.\\associations\\drug_disease.csv')
drug_disease['Label'] = 1
neg = neg_sampling(g, 1, drug['Drug'].values, disease['Disease'].values)
neg['Label'] = 0
data = pd.concat([drug_disease, neg])
data.to_csv('data_1pos_1neg.csv', index=False)
