# %%
import pandas as pd
import lzma
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.model_selection import train_test_split
import time
import os

# %%
# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# %%
# Dataset name
dataset = 'systematic'

# Model parameters
model_type = 'gru'
input_size = 1

# Early stopping parameters
patience = 50
max_epochs = 1000

# %%
# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")

# %%
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

# %%
# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)    # GRU
        self.fc = nn.Linear(hidden_size, 1)                                         # Linear

    def forward(self, x):               
        gru_out, _ = self.gru(x)                            # Pass sequence through GRU    
        last_out = gru_out[:, -1, :]                        # Take the hidden state of the last time step 
        x = self.fc(last_out)                               # Linear combination         
        x = torch.relu(x + 10) - torch.relu(x - 10) - 10    # clamp between -10 and 10
        return x

# %%
# Function to test the model
def test_model(model, inputs):
    model.eval()                                                        # Set model to evaluation mode
    predictions = []

    with torch.no_grad():                                               # Disable gradient calculation
        for seq_input in inputs:
            seq_input = seq_input.unsqueeze(0).unsqueeze(-1).to(device) # Add batch dimension and move to device
            output_seq = model(seq_input)                               # Get model output
            predictions.append(output_seq.item())                       # Store the prediction

    return predictions

# %%
# Function to compute loss value
def get_loss_value(model, test_seqs, y_test, criterion):
    total_test_loss = 0
    with torch.no_grad():                                               # Disable gradient calculation
        for i, seq_input in enumerate(test_seqs):
            target = y_test[i].unsqueeze(0).to(device)                  # Move target to device
            seq_input = seq_input.unsqueeze(0).unsqueeze(-1).to(device) # Prepare input and move to device
            output_seq = model(seq_input)                               # Get model output
            loss = criterion(output_seq, target.unsqueeze(-1))          # Compute loss
            total_test_loss += loss.item()                              # Accumulate loss

    avg_test_loss = total_test_loss / len(test_seqs)                    # Calculate average loss
    return avg_test_loss

# %%
# Load sequence data from CSV
file_path = f'../../sequence_data/{dataset}/profiles.csv.xz'
with lzma.open(file_path, 'rt') as file:
    signal_df = pd.read_csv(file)

# Group sequences by 'sequenceID'
seqs = tuple(signal_df.groupby('sequenceID'))

# Load fold and target data
folds_df = pd.read_csv(f'../../training_data/{dataset}/folds.csv')
target_df = pd.read_csv(f'../../training_data/{dataset}/target.csv')

# %%
# Prepare CSV file for logging
report_path = f'report_{dataset}_{model_type}.csv'
report_header = ['dataset', 'model', 'num_layers', 'hidden_size', 'test_fold', 'stop_epoch', 'train_loss', 'val_loss', 'test_loss', 'time']
if not os.path.exists(report_path):
    pd.DataFrame(columns=report_header).to_csv(report_path, index=False)

# %%
for num_layers in [1, 2]:
    for hidden_size in [2, 4, 8, 16]:
        for test_fold in np.unique(folds_df['fold']):
            # Record start time
            fold_start_time = time.time()

            # Split data into training and test sets based on fold
            train_ids = folds_df[folds_df['fold'] != test_fold]['sequenceID']
            test_ids = folds_df[folds_df['fold'] == test_fold]['sequenceID']

            # Prepare train and test sequences as tensors
            train_seqs = [torch.tensor(seq[1]['signal'].to_numpy(), dtype=torch.float32) for seq in seqs if seq[0] in list(train_ids)]
            test_seqs = [torch.tensor(seq[1]['signal'].to_numpy(), dtype=torch.float32) for seq in seqs if seq[0] in list(test_ids)]

            # Prepare target values for training and testing
            target_df_train = target_df[target_df['sequenceID'].isin(train_ids)]
            y_train = torch.tensor(target_df_train.iloc[:, 1:].to_numpy(), dtype=torch.float32)
            target_df_test = target_df[target_df['sequenceID'].isin(test_ids)]
            y_test = torch.tensor(target_df_test.iloc[:, 1:].to_numpy(), dtype=torch.float32)

            # Split training data into subtrain and validation sets
            train_seqs, val_seqs, y_train, y_val = train_test_split(train_seqs, y_train, test_size=0.2, random_state=42)

            # Initialize the GRU model, loss function, and optimizer
            model = GRUModel(input_size, hidden_size, num_layers).to(device)    # Move model to device
            criterion = SquaredHingeLoss().to(device)                           # Move loss function to device
            optimizer = torch.optim.Adam(model.parameters())

            # Variables for early stopping
            best_train_loss = float('inf')    # Best training loss initialized to infinity
            best_val_loss = float('inf')      # Best validation loss initialized to infinity
            best_test_loss = float('inf')     # Best test loss corresponding to best validation
            patience_counter = 0              # Early stopping patience counter
            best_model_state = None           # Store the best model parameters
            stop_epoch = 0                    # Epoch when training stops

            # Training loop
            for epoch in range(max_epochs):
                # Shuffle training sequences and targets
                combined = list(zip(train_seqs, y_train))
                random.shuffle(combined)
                train_seqs, y_train = zip(*combined)

                total_train_loss = 0
                nan_flag = False  # Flag to detect NaN loss

                # Train on subtrain data
                model.train()  # Set model to training mode
                for i, seq_input in enumerate(train_seqs):
                    target = y_train[i].unsqueeze(0).to(device)  # Prepare target and move to device

                    optimizer.zero_grad()  # Zero gradients

                    # Forward pass
                    seq_input = seq_input.unsqueeze(0).unsqueeze(-1).to(device) # Prepare input and move to device
                    output_seq = model(seq_input)                               # Get model output
                    loss = criterion(output_seq, target.unsqueeze(-1))          # Compute loss

                    if torch.isnan(loss).any():  # Check for NaN loss
                        print(f"NaN loss detected at Epoch [{epoch}], Step [{i}]")
                        nan_flag = True
                        break

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()  # Accumulate training loss

                if nan_flag:
                    break  # Stop training if NaN was encountered

                # Calculate average training loss
                avg_train_loss = total_train_loss / len(train_seqs)

                # Calculate validation and test losses
                avg_val_loss = get_loss_value(model, val_seqs, y_val, criterion)
                avg_test_loss = get_loss_value(model, test_seqs, y_test, criterion)

                if epoch % 20 == 0:
                    print(f'Test fold {test_fold} \t Epoch [{epoch:3d}] \t Avg Train Loss: {avg_train_loss:.8f} \t Avg Val Loss: {avg_val_loss:.8f} \t Avg Test Loss: {avg_test_loss:.8f}')

                # Early stopping based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss        # Update best validation loss
                    best_train_loss = avg_train_loss    # Store best training loss
                    best_test_loss = avg_test_loss      # Store test loss for best validation
                    patience_counter = 0                # Reset patience counter

                    # Save best model parameters
                    best_model_state = model.state_dict()
                    stop_epoch = epoch + 1              # Record stopping epoch
                else:
                    patience_counter += 1               # Increment patience counter

                # Stop training if patience is exceeded
                if patience_counter > patience:
                    print(f"Test fold {test_fold} \t Early stopping at Epoch [{epoch}]")
                    break

            # Record total time taken for this fold
            fold_duration = time.time() - fold_start_time

            # Save results to CSV
            report_entry = {
                'dataset': dataset,
                'model': model_type,
                'num_layers': num_layers,
                'hidden_size': hidden_size,
                'test_fold': test_fold,
                'stop_epoch': stop_epoch,
                'train_loss': best_train_loss,
                'val_loss': best_val_loss,
                'test_loss': best_test_loss,
                'time': fold_duration
            }

            pd.DataFrame([report_entry]).to_csv(report_path, mode='a', header=False, index=False)  # Append entry to CSV

            print(f"Test fold {test_fold} \t Training completed for GRU layers {num_layers} \t Hidden size {hidden_size} \t Best Val Loss: {best_val_loss:.8f} \t Best Test Loss: {best_test_loss:.8f}")
            
            # Restore best model parameters after training
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                model.eval()  # Set the model to evaluation mode

            # Test the model and collect outputs
            pred_lldas = test_model(model, test_seqs)

            # Save model parameters
            torch.save(model.state_dict(), f'saved_models/{model_type}_{dataset}_{num_layers}layers_{hidden_size}features_fold{test_fold}.pth')

            # Save predictions to CSV
            lldas_df = pd.DataFrame(list(zip(test_ids, pred_lldas)), columns=['sequenceID', 'llda'])
            lldas_df.to_csv(f'predictions/{model_type}_{dataset}_{num_layers}layers_{hidden_size}features_fold{test_fold}.csv', index=False)


