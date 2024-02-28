import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset
from utility_functions import gen_data_dict, get_data, SquaredHingeLoss

# Set random seeds for reproducibility
np.random.seed(4)
torch.manual_seed(4)

# Constants and hyperparameters
CHOSEN_FEATURES = ['std_deviation', 'count', 'sum_diff', 'range_value', 'abs_skewness']
CV_BATCH_SIZE = 16
CV_N_FOLDS = 2
CV_N_ITE = 200
TRAIN_BATCH_SIZE = 1
LR = 0.001
N_HIDDEN_VALUES = [1, 2, 3]
LAYER_SIZE_VALUES = [4, 8, 16]
CONFIGS = [{'n_hiddens': 0, 'layer_size': 0}]
CONFIGS += [{'n_hiddens': n, 'layer_size': s} for n in N_HIDDEN_VALUES for s in LAYER_SIZE_VALUES]

# Load error data and sequences/labels data
ERR_FOLD1_DF = pd.read_csv('1_genome/1_training_data/errors_fold1_base_10.csv')
ERR_FOLD2_DF = pd.read_csv('1_genome/1_training_data/errors_fold2_base_10.csv')
SEQS = gen_data_dict('1_genome/0_sequences_labels/signals.gz')
LABELS = gen_data_dict('1_genome/0_sequences_labels/labels.gz')

# Load and preprocess features
def load_and_preprocess_features():
    X = pd.read_csv('1_genome/1_training_data/seq_features.csv').iloc[:, 1:][CHOSEN_FEATURES].to_numpy()
    X = np.log(X)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    features = torch.Tensor(X)
    return features

# Load target data
def load_target_data():
    target_df_1 = pd.read_csv('1_genome/1_training_data/target_lambda_fold1_base_10.csv')
    target_df_2 = pd.read_csv('1_genome/1_training_data/target_lambda_fold2_base_10.csv')
    targets_low_1 = torch.Tensor(target_df_1.iloc[:, 1:2].to_numpy())
    targets_high_1 = torch.Tensor(target_df_1.iloc[:, 2:3].to_numpy())
    targets_low_2 = torch.Tensor(target_df_2.iloc[:, 1:2].to_numpy())
    targets_high_2 = torch.Tensor(target_df_2.iloc[:, 2:3].to_numpy())
    target_fold1 = torch.cat((targets_low_1, targets_high_1), dim=1)
    target_fold2 = torch.cat((targets_low_2, targets_high_2), dim=1)
    return target_fold1, target_fold2

# Define the deep learning model
class DLModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size):
        super(DLModel, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        if self.hidden_layers == 0:
            self.linear_model = nn.Linear(input_size, 1)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)
            self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers - 1)])
            self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.hidden_layers == 0:
            return self.linear_model(x)
        else:
            x = torch.relu(self.input_layer(x))
            for layer in self.hidden:
                x = torch.relu(layer(x))
            x = self.output_layer(x)
            return x

# Cross-validation function
def cv_learn(n_splits, X, y, n_hiddens, layer_size, lr, n_ite):
    kf = KFold(n_splits, shuffle=True, random_state=0)
    loss_func = SquaredHingeLoss(margin=1)
    total_train_losses = np.zeros(n_ite)
    total_val_losses = np.zeros(n_ite)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=CV_BATCH_SIZE, shuffle=True)
        model = DLModel(len(CHOSEN_FEATURES), n_hiddens, layer_size)
        optimizer = optim.Adam(model.parameters(), lr)
        train_losses = []
        val_losses = []
        for i in range(n_ite):
            train_loss = 0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                loss = loss_func(model(inputs), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            with torch.no_grad():
                val_loss = loss_func(model(X_val), y_val)
            train_losses.append(train_loss / len(dataloader))
            val_losses.append(val_loss.item())
        total_train_losses += train_losses
        total_val_losses += val_losses
    best_no_ite = np.argmin(total_val_losses)
    return best_no_ite + 1

# Training function
def train_model(X, y, n_hiddens, layer_size, lr, n_ites):
    model = DLModel(len(CHOSEN_FEATURES), n_hiddens, layer_size)
    loss_func = SquaredHingeLoss(margin=1)
    optimizer = optim.Adam(model.parameters(), lr)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    for _ in range(n_ites):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            loss = loss_func(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model

# Function to calculate statistics and errors
def get_df_stat(ldas1, ldas2, err_fold1_df, err_fold2_df, seqs, labels):
    header = ['sequenceID', 'lda_fold1', 'lda_fold2', 'fold_1_total_labels', 'fold_2_total_labels', 'fold1_err', 'fold2_err']
    rows = []
    for i in range(len(seqs)):
        _, neg_start_1, _, pos_start_1, _, neg_start_2, _, pos_start_2, _ = get_data(i, seqs, labels)
        fold1_total_labels = len(neg_start_1) + len(pos_start_1)
        fold2_total_labels = len(neg_start_2) + len(pos_start_2)
        ldas1 = [round(num * 2) / 2 for num in ldas1]
        ldas2 = [round(num * 2) / 2 for num in ldas2]
        fold1_err = err_fold1_df.iloc[i][str(ldas1[i])]
        fold2_err = err_fold2_df.iloc[i][str(ldas2[i])]
        row = [seqs[i][0], ldas1[i], ldas2[i], fold1_total_labels, fold2_total_labels, fold1_err, fold2_err]
        rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    return df

# Function to try different model configurations
def try_model(full_X, X1, X2, y1, y2, config, err_fold1_df, err_fold2_df, seqs, labels):
    n_hiddens = config['n_hiddens']
    layer_size = config['layer_size']
    best_no_ite_1 = cv_learn(CV_N_FOLDS, X1, y1, n_hiddens, layer_size, LR, CV_N_ITE)
    best_no_ite_2 = cv_learn(CV_N_FOLDS, X2, y2, n_hiddens, layer_size, LR, CV_N_ITE)
    model1 = train_model(X1, y1, n_hiddens, layer_size, LR, best_no_ite_1)
    model2 = train_model(X2, y2, n_hiddens, layer_size, LR, best_no_ite_2)
    with torch.no_grad():
        ldas1 = model1(full_X).numpy().reshape(-1)
        ldas2 = model2(full_X).numpy().reshape(-1)
    df = get_df_stat(ldas1, ldas2, err_fold1_df, err_fold2_df, seqs, labels)
    return df

def main():
    # Load and preprocess features
    features = load_and_preprocess_features()

    # Load target data
    target_fold1, target_fold2 = load_target_data()

    # Parallel execution of model trials
    dfs = Parallel(n_jobs=len(CONFIGS), verbose=1)(
        delayed(try_model)(features, features, features, target_fold1, target_fold2,
                           CONFIGS[i], ERR_FOLD1_DF, ERR_FOLD2_DF, SEQS, LABELS) for i in range(len(CONFIGS)))

    # Processing results
    results = []
    for i in range(len(dfs)):
        df = dfs[i]
        total_label_fold1 = df['fold_1_total_labels'].sum()
        total_label_fold2 = df['fold_2_total_labels'].sum()
        err1 = df['fold1_err'].sum()
        err2 = df['fold2_err'].sum()
        rate1 = (total_label_fold1 - err1) / total_label_fold1
        rate2 = (total_label_fold2 - err2) / total_label_fold2
        results.append({
            'n_hiddens': CONFIGS[i]['n_hiddens'],
            'layer_size': CONFIGS[i]['layer_size'],
            'fold1_test': rate1 * 100,
            'fold2_test': rate2 * 100
        })

    df_results = pd.DataFrame(results)
    print(df_results)

    # Calculate and print average test results
    fold1_test_avg = df_results['fold1_test'].mean()
    fold2_test_avg = df_results['fold2_test'].mean()
    print("Average fold1_test: %.2f" % fold1_test_avg)
    print("Average fold2_test: %.2f" % fold2_test_avg)

if __name__ == "__main__":
    main()