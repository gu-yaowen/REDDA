import torch as th
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F


class HeteroPredictor(nn.Module):
    def __init__(self, in_dim, predictor='MLP'):
        super().__init__()
        if predictor == 'Linear':
            self.predictor = nn.Linear(in_dim, 1)
        elif predictor == 'MLP':
            self.predictor = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim // 2, in_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim // 4, 1)
            )

    def forward(self, h, drug_id, dis_id):
        drug_h = h['drug'][drug_id]
        disease_h = h['disease'][dis_id]
        return self.predictor(th.cat((drug_h, disease_h), dim=-1))


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, predictor):
        super().__init__()
        self.linear_drug = nn.Linear(2048, in_features)
        self.linear_protein = nn.Linear(1280, in_features)
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroPredictor(out_features * 2, predictor=predictor)

    def forward(self, g, x, drug_id, dis_id):
        h = {'drug': self.linear_drug(x['drug']),
        'protein': self.linear_protein(x['protein'])}
        # h = {'drug': self.linear_drug(x['drug'])}
        # h = {'protein': self.linear_protein(x['protein'])}
        h = self.sage(g, h)
        return self.pred(h, drug_id, dis_id)
