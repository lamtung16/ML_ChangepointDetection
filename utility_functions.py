
import numpy as np
import pandas as pd
import csv
import os
import torch
import torch.nn as nn



# get number of error from sequenceID and llda
def get_err(evaluation_df, seqID, llda):
    # get sub eval_df of seqID
    eval_df = evaluation_df[evaluation_df['sequenceID'] == seqID]
    
    # get right row
    position = np.logical_and(eval_df['min.log.lambda'] < llda, llda < eval_df['max.log.lambda'])
    row = eval_df[position]

    # get total labels and total errors
    n_labels = row['labels'].item()
    n_errs = row['errors'].item()

    return n_labels, n_errs



# add row to csv
def add_row_to_csv(path, head, row):
    file_exists = os.path.exists(path)              # Check if the file exists
    is_row_exist = False                            # default False for is_row_exist
    with open(path, 'a', newline='') as csvfile:    # Open the CSV file in append mode
        writer = csv.writer(csvfile)
        if not file_exists:                         # If the file doesn't exist, write the header
            writer.writerow(head)
        writer.writerow(row)                        # Write the row




# get acc from eval_df and lldas_df
def get_acc(eval_df, lldas_df):
    total_err = 0
    total_labels = 0
    for seqID in lldas_df['sequenceID']:
        llda = lldas_df[lldas_df['sequenceID'] == seqID]['llda'].item()
        n_labels, n_errs = get_err(eval_df, seqID, llda)
        total_labels += n_labels
        total_err += n_errs
    acc = (total_labels - total_err)/total_labels
    return acc*100




# Hinged Square Loss
class SquaredHingeLoss(nn.Module):
    def __init__(self, margin=1):
        super(SquaredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, y):
        low, high = y[:, 0:1], y[:, 1:2]
        loss_low = torch.relu(low - predicted + self.margin)
        loss_high = torch.relu(predicted - high + self.margin)
        loss = loss_low + loss_high
        return torch.mean(torch.square(loss))
