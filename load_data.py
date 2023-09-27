import dgl
import torch as th
import numpy as np
import pandas as pd


def load(dataset):
    """Load the heterogeneous network of Bdataset or Kdataset.
       Note: It is also applicable to load other heterogeneous networks of your own datasets.
    """

    if dataset == 'Bdataset':
        return load_Bdataset()
    if dataset == 'Kdataset':
        return load_Kdataset()

def load_Kdataset():
    """Load the heterogeneous network of Kdataset.
    """

    drug_drug = pd.read_csv('./dataset/Kdataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/Kdataset/interactions/protein_protein.csv')
    gene_gene = pd.read_csv('./dataset/Kdataset/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv('./dataset/Kdataset/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv('./dataset/Kdataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/Kdataset/associations/drug_protein.csv')
    protein_gene = pd.read_csv('./dataset/Kdataset/associations/protein_gene.csv')
    gene_pathway = pd.read_csv('./dataset/Kdataset/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv('./dataset/Kdataset/associations/pathway_disease.csv')
    drug_disease = pd.read_csv('./dataset/Kdataset/associations/Kdataset.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
                                              th.tensor(protein_gene['Gene'].values)),
        ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
                                              th.tensor(protein_gene['Protein'].values)),
        ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
                                        th.tensor(gene_gene['Gene2'].values)),
        ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
                                              th.tensor(gene_pathway['Pathway'].values)),
        ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
                                              th.tensor(gene_pathway['Gene'].values)),
        ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
                                                    th.tensor(pathway_pathway['Pathway2'].values)),
        ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
                                                    th.tensor(pathway_disease['Disease'].values)),
        ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
                                                    th.tensor(pathway_disease['Pathway'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)
    g.nodes['gene'].data['h'] = th.zeros((g.num_nodes('gene'), drug_feature.shape[1])).to(th.float32)
    g.nodes['pathway'].data['h'] = th.zeros((g.num_nodes('pathway'), drug_feature.shape[1])).to(th.float32)
    return g

def load_Bdataset():
    """Load the heterogeneous network of Bdataset.
    """

    drug_drug = pd.read_csv('./dataset/Bdataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/Bdataset/interactions/protein_protein.csv')
    disease_disease = pd.read_csv('./dataset/Bdataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/Bdataset/associations/drug_protein.csv')
    drug_disease = pd.read_csv('./dataset/Bdataset/associations/Bdataset.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), g.num_nodes('protein'))).to(th.float32)
    return g  

def remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.tensor(test_drug_id),
                          th.tensor(test_dis_id),
                          etype=('drug', 'drug_disease', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'drug_disease', 'disease'))
    edges_id = g.edge_ids(th.tensor(test_dis_id),
                          th.tensor(test_drug_id),
                          etype=('disease', 'disease_drug', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('disease', 'disease_drug', 'drug'))
    return g


# def construct_neg_graph(g, test_id, k_drug, k_dis):
#     test_drug_id = test_id[:, 0]
#     test_dis_id = test_id[:, 1]
#     neg_drug = th.randint(0, g.num_nodes('drug'), (len(test_dis_id) * k_drug,))
#     neg_drug_dis = th.tensor(np.array([test_dis_id.tolist() for i in range(k_drug)]).flatten())
#     neg_dis = th.randint(0, g.num_nodes('disease'), (len(test_drug_id) * k_dis,))
#     neg_dis_drug = th.tensor(np.array([test_drug_id.tolist() for i in range(k_dis)]).flatten())
#     g.add_edges(neg_dis_drug, neg_dis,
#                 etype=('drug', 'drug_disease', 'disease'))
#     g.add_edges(neg_drug, neg_drug_dis,
#                 etype=('drug', 'drug_disease', 'disease'))
#     g.add_edges(neg_dis, neg_dis_drug,
#                 etype=('disease', 'disease_drug', 'drug'))
#     g.add_edges(neg_drug_dis, neg_drug,
#                 etype=('disease', 'disease_drug', 'drug'))
#     return g
