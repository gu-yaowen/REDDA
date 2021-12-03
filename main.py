
import os
import numpy as np
import pandas as pd
import dgl
import torch as th
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from sklearn.model_selection import KFold
from load_data import load, remove_edge
from utils import cal_metric, set_seed, plot_result
from args import args


def train():
    print(args)
    set_seed(args.seed)
    try:
        os.mkdir(args.saved_path)
    except:
        pass
    if args.device_id:
        th.cuda.set_device(args.device_id)
    th.set_default_tensor_type(th.FloatTensor)
    data = pd.read_csv('.\\dataset\\'+args.dataset+'.csv')
    data = data.astype('int64')
    data = data.sample(frac=1).reset_index(drop=True).values
    criterion = th.nn.BCEWithLogitsLoss()
    pred_list = np.zeros(len(data))
    loss_list, val_list = [], []
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    fold = 1
    for train_idx, test_idx in kf.split(data):
        print('5-Cross Validation: Fold {}'.format(fold))
        train_drug_id, train_dis_id = data[train_idx, 0], data[train_idx, 1]
        train_label = th.tensor(data[train_idx, -1]).float()
        test_drug_id, test_dis_id = data[test_idx, 0], data[test_idx, 1]
        test_label = th.tensor(data[test_idx, -1])

        g = load()
        g = remove_edge(g, test_drug_id, test_dis_id, test_label)
        if args.device_id:
            g.to('cuda:{}'.format(args.device_id))
        node_features = {'drug': g.nodes['drug'].data['h'],
                         'protein': g.nodes['protein'].data['h']}
        # node_features = {'drug': g.nodes['drug'].data['h']}
        # node_features = {'protein': g.nodes['protein'].data['h']}
        model = Model(1000, 2000, 200, g.etypes, 'MLP')
        if args.device_id:
            model.cuda(device=args.device_id)
        opt = th.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_, val_ = [], []

        for epoch in range(args.epoch):
            model.train()
            score = model(g, node_features, train_drug_id, train_dis_id).cpu()
            AUC_, _, _, _ = cal_metric(train_label.long(), score.detach().numpy())
            loss = criterion(score.squeeze(), train_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.eval()
            pred = th.sigmoid(model(g, node_features, test_drug_id, test_dis_id)).detach().numpy()
            AUC, aupr, acc, f1 = cal_metric(test_label, pred)
            print(
                'Epoch {} Loss: {:.3f}; Train AUC {:.3f}; Val AUC: {:.3f}; Val AUPR: {:.3f}; Val Acc: {:.3f}; Val F1: {:.3f}'.
                    format(epoch, loss.item(), AUC_, AUC, aupr, acc, f1))
            loss_.append(loss.item())
            val_.append(AUC)
        loss_list.append(loss_)
        val_list.append(val_)
        pred_list[test_idx] = pred.squeeze()
        print('-'*30+'+'*30+'-'*30)
        th.save(model.state_dict(), os.path.join(args.saved_path, 'model_{}.pth'.format(fold)))
        fold += 1

    pd.DataFrame(np.array(loss_list).T,
                 columns=['Fold_{}'.format(i)
                          for i in range(5)]).to_csv(os.path.join(args.saved_path, 'loss.csv'), index=False)
    pd.DataFrame(np.array(val_list).T,
                 columns=['Fold_{}'.format(i)
                          for i in range(5)]).to_csv(os.path.join(args.saved_path, 'val.csv'), index=False)
    pd.DataFrame(np.array([data[:, -1], pred_list]).T,
                 columns=['Label', 'Predict']).to_csv(os.path.join(args.saved_path, 'pred.csv'), index=False)
    plot_result(args, data[:, -1], pred_list)


if __name__ == '__main__':
    train()

