import dgl
import torch as th
import numpy as np
import pandas as pd


def load():
    drug_drug = pd.read_csv('./dataset/interactions/drug_drug.csv')
    protein_protein = pd.read_csv('./dataset/interactions/protein_protein.csv')
    gene_gene = pd.read_csv('./dataset/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv('./dataset/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv('./dataset/interactions/disease_disease.csv')
    drug_protein = pd.read_csv('./dataset/associations/drug_protein.csv')
    protein_gene = pd.read_csv('./dataset/associations/protein_gene.csv')
    gene_pathway = pd.read_csv('./dataset/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv('./dataset/associations/pathway_disease.csv')
    drug_disease = pd.read_csv('./dataset/associations/drug_disease.csv')
    drug = pd.read_csv('./dataset/omics/drug.csv.zip', compression='zip')
    protein = pd.read_csv('./dataset/omics/protein.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        # ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
        #                                       th.tensor(drug_protein['Protein'].values)),
        # ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
        #                                             th.tensor(protein_protein['Protein2'].values)),
        # ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
        #                                       th.tensor(protein_gene['Gene'].values)),
        # ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
        #                                 th.tensor(gene_gene['Gene2'].values)),
        # ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
        #                                       th.tensor(gene_pathway['Pathway'].values)),
        # ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
        #                                             th.tensor(pathway_pathway['Pathway2'].values)),
        # ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
        #                                             th.tensor(pathway_disease['Disease'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values))
    }
    g = dgl.heterograph(graph_data)
    drug_fea = np.array([[float(j) for j in i.replace(' ', '')[1:-1].split(',')]
                         for i in drug['Fingerprint'].values])
    # protein_fea = np.array([[float(j) for j in i.replace(' ', '')[1:-1].split(',')]
    #                         for i in protein['Feature'].values])
    g.nodes['drug'].data['h'] = th.from_numpy(drug_fea).float()
    # g.nodes['protein'].data['h'] = th.from_numpy(protein_fea).float()
    g.edges['drug_drug'].data['h_e'] = th.from_numpy(drug_drug['Sim'].values).float()
    g.edges['disease_disease'].data['h_e'] = th.from_numpy(disease_disease['Sim'].values).float()
    return g


def remove_edge(g, test_drug_id, test_dis_id, test_label):
    edges_id = g.edge_ids(th.tensor(test_drug_id[np.where(test_label == 1)[0]]),
                          th.tensor(test_dis_id[np.where(test_label == 1)[0]]),
                          etype=('drug', 'drug_disease', 'disease'))
    return dgl.remove_edges(g, edges_id, etype=('drug', 'drug_disease', 'disease'))
