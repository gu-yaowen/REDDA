import random
import seaborn
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc,\
    precision_recall_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt


def cal_metric(label, pred):
    AUC = roc_auc_score(label, pred)
    precision, recall, thresholds = precision_recall_curve(label, pred)
    aupr = auc(recall, precision)
    pred_binary = np.array([0 if i < 0.5 else 1 for i in pred])
    acc = accuracy_score(label, pred_binary)
    f1 = f1_score(label, pred_binary)
    return AUC, aupr, acc, f1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def plot_result(args, label, predict):
    seaborn.set_style()
    fpr, tpr, threshold = roc_curve(label, predict)
    score = roc_auc_score(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result.png'))
    plt.clf()

