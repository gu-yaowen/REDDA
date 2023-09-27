import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim=None, dropout=0.4):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, feature):
        feature['drug'] = self.dropout(feature['drug'])
        feature['disease'] = self.dropout(feature['disease'])
        R = feature['drug']
        D = feature['disease']
        D = self.weights(D)
        outputs = R @ D.T
        return outputs


class Node_Embedding(nn.Module):
    """The base HeteroGCN layer."""

    def __init__(self, in_feats, out_feats, dropout, rel_names):
        super().__init__()
        HeteroGraphdict = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(in_feats, out_feats)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict[rel] = graphconv
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = dglnn.HeteroGraphConv(HeteroGraphdict, aggregate='sum')
        self.bn_layer = nn.BatchNorm1d(out_feats)
        self.prelu = nn.PReLU()

    def forward(self, graph, inputs, bn=False, dp=False):
        h = self.embedding(graph, inputs)
        if bn and dp:
            h = {k: self.prelu(self.dropout(self.bn_layer(v))) for k, v in h.items()}
        elif dp:
            h = {k: self.prelu(self.dropout(v)) for k, v in h.items()}
        elif bn:
            h = {k: self.prelu(self.bn_layer(v)) for k, v in h.items()}
        else:
            h = {k: self.prelu(v) for k, v in h.items()}
        return h


class SemanticAttention(nn.Module):
    """The base attention mechanism used in
    topological subnet embedding block and layer attention block.
    """

    def __init__(self, in_feats, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, is_print=False):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        if is_print:
            print(beta)
        return (beta * z).sum(1)


class SubnetworkEncoder(nn.Module):
    """Topological subnet embedding block."""

    def __init__(self, ntypes, in_feats, out_feats, dropout):
        super(SubnetworkEncoder, self).__init__()
        self.ntypes = ntypes
        self.drug_disease = Node_Embedding(in_feats, out_feats, dropout,
                                            ['drug_drug', 'drug_disease', 'disease_disease'])
        if 'drug' in ntypes and 'protein' in ntypes:
            self.drug_protein = Node_Embedding(in_feats, out_feats, dropout,
                                         ['drug_drug', 'drug_protein', 'protein_protein'])
        if 'protein' in ntypes and 'gene' in ntypes:
            self.protein_gene = Node_Embedding(in_feats, out_feats, dropout,
                                               ['protein_protein', 'protein_gene', 'gene_gene'])
        if 'gene' in ntypes and 'pathway' in ntypes:
            self.gene_pathway = Node_Embedding(in_feats, out_feats, dropout,
                                               ['gene_gene', 'gene_pathway', 'pathway_pathway'])
        if 'pathway' in ntypes and 'disease' in ntypes:
            self.pathway_disease = Node_Embedding(in_feats, out_feats, dropout,
                                                  ['pathway_pathway', 'pathway_disease', 'disease_disease'])
        self.semantic_attention = SemanticAttention(in_feats=out_feats)

    def forward(self, g, h, bn=False, dp=False):
        new_h = {}
        for ntype in self.ntypes:
            new_h[ntype] = []

        # drug-disease subnet
        g_ = g.edge_type_subgraph(['drug_drug', 'drug_disease', 'disease_disease'])
        h_ = self.drug_disease(g_, {'drug': h['drug'], 'disease': h['disease']}, bn, dp)
        new_h['drug'].append(h_['drug'])
        new_h['disease'].append(h_['disease'])

        # drug-protein subnet
        if 'drug' in self.ntypes and 'protein' in self.ntypes:
            g_ = g.edge_type_subgraph(['drug_drug', 'drug_protein', 'protein_protein'])
            h_ = self.drug_protein(g_, {'drug': h['drug'], 'protein': h['protein']}, bn, dp)
            new_h['drug'].append(h_['drug'])
            new_h['protein'].append(h_['protein'])

        # protein-gene subnet
        if 'protein' in self.ntypes and 'gene' in self.ntypes:
            g_ = g.edge_type_subgraph(['protein_protein', 'protein_gene', 'gene_gene'])
            h_ = self.protein_gene(g_, {'protein': h['protein'], 'gene': h['gene']}, bn, dp)
            new_h['protein'].append(h_['protein'])
            new_h['gene'].append(h_['gene'])

        # gene-pathway subnet
        if 'gene' in self.ntypes and 'pathway' in self.ntypes:
            g_ = g.edge_type_subgraph(['gene_gene', 'gene_pathway', 'pathway_pathway'])
            h_ = self.gene_pathway(g_, {'gene': h['gene'], 'pathway': h['pathway']}, bn, dp)
            new_h['gene'].append(h_['gene'])
            new_h['pathway'].append(h_['pathway'])

        # pathway-disease subnet
        if 'pathway' in self.ntypes and 'disease' in self.ntypes:
            g_ = g.edge_type_subgraph(['pathway_pathway', 'pathway_disease', 'disease_disease'])
            h_ = self.pathway_disease(g_, {'pathway': h['pathway'], 'disease': h['disease']}, bn, dp)
            new_h['pathway'].append(h_['pathway'])
            new_h['disease'].append(h_['disease'])

        # aggragation with attention mechanism
        for ntype in self.ntypes:
            h[ntype] = torch.stack(new_h[ntype], dim=1)
            h[ntype] = self.semantic_attention(h[ntype])
        return h


class Graph_attention(nn.Module):
    """Multi-omics graph attention block."""

    def __init__(self, in_feats, out_feats, num_heads, dropout):
        super().__init__()
        self.gat = dglnn.GATConv(in_feats, out_feats, num_heads,
                                 dropout, dropout,
                                 activation=nn.PReLU(),
                                 allow_zero_in_degree=True)
        self.gat.reset_parameters()
        self.linear = nn.Linear(in_feats * num_heads, out_feats)
        self.prelu = nn.PReLU()
        self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, graph, inputs, bn=False):
        # disease drug gene pathway protein
        num_dis = graph.num_nodes('disease')
        num_drug = graph.num_nodes('drug')
        new_g = dgl.to_homogeneous(graph)
        new_h = torch.cat([i for i in inputs.values()], dim=0)
        new_h = self.gat(new_g, new_h)
        new_h = self.prelu(torch.mean(new_h, dim=1))
        if bn:
            return self.bn_layer(new_h[:num_dis]), self.bn_layer(new_h[num_dis:num_drug + num_dis])
        return new_h[:num_dis], new_h[num_dis:num_drug + num_dis]


class Model(nn.Module):
    """The overall MODDA architecture."""

    def __init__(self, etypes, ntypes, in_feats, hidden_feats, num_heads, dropout):
        super(Model, self).__init__()
        self.ntypes = ntypes
        if 'drug' in ntypes:
            self.drug_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.drug_linear.weight)
        if 'disease' in ntypes:
            self.disease_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.disease_linear.weight)
        if 'protein' in ntypes:
            self.protein_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.protein_linear.weight)
        if 'gene' in ntypes:
            self.gene_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.gene_linear.weight)
        if 'pathway' in ntypes:
            self.pathway_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.pathway_linear.weight)

        self.feat_generate_layer1 = Node_Embedding(hidden_feats, hidden_feats, dropout, etypes)
        self.feat_generate_layer2 = Node_Embedding(hidden_feats, hidden_feats, dropout, etypes)
        self.subnet_layer = SubnetworkEncoder(ntypes, hidden_feats, hidden_feats, dropout)
        self.totalnet_layer = Graph_attention(hidden_feats, hidden_feats, num_heads, dropout)
        self.layer_attention_layer_drug = SemanticAttention(hidden_feats)
        self.layer_attention_layer_dis = SemanticAttention(hidden_feats)
        self.predict = InnerProductDecoder(hidden_feats)

    def forward(self, g, x):
        drug_emb_list, dis_emb_list = [], []

        h = {}
        for ntype in self.ntypes:
            h[ntype] = x[ntype]
        h['drug'] = self.drug_linear(h['drug'])
        h['disease'] = self.disease_linear(h['disease'])
        if 'protein' in self.ntypes:
            h['protein'] = self.protein_linear(h['protein'])
        if 'gene' in self.ntypes:
            h['gene'] = self.gene_linear(h['gene'])
        if 'pathway' in self.ntypes:
            h['pathway'] = self.pathway_linear(h['pathway'])
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])

        h = self.feat_generate_layer1(g, h, bn=True, dp=True)
        # drug_emb_list.append(h['drug'])
        # dis_emb_list.append(h['disease'])
        h = self.feat_generate_layer2(g, h, bn=True, dp=True)
        # h['drug'] = torch.cat((drug_emb_list[0], h['drug']), dim=0)
        # h['disease'] = torch.cat((dis_emb_list[0], h['disease']), dim=0)
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])

        h = self.subnet_layer(g, h, bn=False, dp=True)
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])

        h['disease'], h['drug'] = self.totalnet_layer(g, h, bn=False)
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])

        # h['drug'] = torch.cat(drug_emb_list, dim=1)
        # h['disease'] = torch.cat(dis_emb_list, dim=1)
        h['drug'] = self.layer_attention_layer_drug(torch.stack(drug_emb_list, dim=1))
        h['disease'] = self.layer_attention_layer_dis(torch.stack(dis_emb_list, dim=1))

        return self.predict(h)
