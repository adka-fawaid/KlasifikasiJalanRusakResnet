# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct, preds

def compute_confusion(true_all, pred_all, labels):
    cm = confusion_matrix(true_all, pred_all, labels=labels)
    return cm

def classification_report_str(true_all, pred_all, target_names):
    return classification_report(true_all, pred_all, target_names=target_names, digits=4)
