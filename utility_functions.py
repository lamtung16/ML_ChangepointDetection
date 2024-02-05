import numpy as np
import pandas as pd
import gzip
import csv
import os
import torch
import torch.nn as nn


# Get cumulative sum vectors
def get_cumsum(sequence):
    y = np.cumsum(sequence)
    z = np.cumsum(np.square(sequence))
    return np.append([0], y), np.append([0], z)


# function to create loss value from 'start' to 'end' given cumulative sum vector y (data) and z (square)
def L(start, end, y, z):
    _y = y[end+1] - y[start]
    _z = z[end+1] - z[start]
    return _z - np.square(_y)/(end-start+1)


# function to get the list of changepoint from vector tau_star
def trace_back(tau_star):
    tau = tau_star[-1]
    chpnt = np.array([len(tau_star)], dtype=int)
    while tau > 0:
        chpnt = np.append(tau, chpnt)
        tau = tau_star[tau-1]
    return np.append(0, chpnt)


# counting errors
def error_count(chpnt, neg_start, neg_end, pos_start, pos_end):
    chpnt = chpnt[1:]
    fp = 0
    fn = 0
    for ns, ne in zip(neg_start, neg_end):
        count = sum(1 for cp in chpnt if ns <= cp < ne)
        if count >= 1:
            fp += 1
    for ps, pe in zip(pos_start, pos_end):
        count = sum(1 for cp in chpnt if ps <= cp < pe)
        if count >= 2:
            fp += 1
        elif count == 0:
            fn += 1
    return fp, fn


# OPART given lambda and sequence
def opart(lda, sequence):
    y, z = get_cumsum(sequence)             # cumsum vector
    sequence_length = len(sequence)-1       # length of sequence

    # Set up
    C = np.zeros(sequence_length + 1)
    C[0] = -lda

    # Get tau_star
    tau_star = np.zeros(sequence_length+1, dtype=int)
    for t in range(1, sequence_length+1):
        V = C[:t] + lda + L(1 + np.arange(t), t, y, z)  # calculate set V
        last_chpnt = np.argmin(V)                       # get optimal tau from set V
        C[t] = V[last_chpnt]                            # update C_i
        tau_star[t] = last_chpnt                        # update tau_star

    set_of_chpnt = trace_back(tau_star[1:])             # get set of changepoints
    return set_of_chpnt


# Loss function
class SquaredHingeLoss(nn.Module):
    def __init__(self, margin=1):
        super(SquaredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, y):
        low, high = y[:, 0], y[:, 1]
        margin = self.margin
        loss_low = torch.relu(low - predicted + margin)
        loss_high = torch.relu(predicted - high + margin)
        loss = loss_low + loss_high
        return torch.mean(torch.square(loss))


# Show error
def show_error_rate(df):
    fold1_total_errs = df['fold_1_fp_errs'].sum() + df['fold_1_fn_errs'].sum()
    fold2_total_errs = df['fold_2_fp_errs'].sum() + df['fold_2_fn_errs'].sum()

    fold1_total_labels = df['fold_1_total_labels'].sum()
    fold2_total_labels = df['fold_2_total_labels'].sum()

    rate1 = 100*(fold1_total_labels - fold1_total_errs)/fold1_total_labels
    rate2 = 100*(fold2_total_labels - fold2_total_errs)/fold2_total_labels

    return rate1, rate2, fold1_total_labels, fold2_total_labels, fold1_total_errs, fold2_total_errs