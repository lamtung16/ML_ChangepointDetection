import numpy as np
import pandas as pd

def gen_data_dict(file_path):
    df = pd.read_csv(file_path, compression='xz')
    df['index'] = df.groupby('sequenceID').cumcount() + 1
    _dict = tuple(df.groupby('sequenceID'))
    return _dict


def find_closest_index(df, arr_pos):
    # Sort the DataFrame by 'position_a' to facilitate finding the closest index
    df_sorted = df.sort_values(by='position')
    
    # Function to find the closest index for a given position
    def find_closest(position):
        closest_index = None
        min_distance = np.inf
        
        for i, row in df_sorted.iterrows():
            distance = abs(row['position'] - position)
            if distance < min_distance:
                min_distance = distance
                closest_index = row['index']
        
        return closest_index
    
    # Find closest index for positions in array B
    closest_indices = [find_closest(position) for position in arr_pos]
    return closest_indices


def get_data(i, seqs, labels):
    # sequence
    sequence = seqs[i][1]['signal'].to_numpy()
    sequence = np.append([0], sequence)

    # labels
    lab_df = labels[i][1]

    # get label sets
    neg_start = lab_df[lab_df['max.changes'] == 0.0]['labelStart'].to_numpy()
    neg_end   = lab_df[lab_df['max.changes'] == 0.0]['labelEnd'].to_numpy()
    pos_start = lab_df[lab_df['max.changes'] == 1.0]['labelStart'].to_numpy()
    pos_end   = lab_df[lab_df['max.changes'] == 1.0]['labelEnd'].to_numpy()

    neg_start = find_closest_index(seqs[i][1], neg_start)
    neg_end   = find_closest_index(seqs[i][1], neg_end)
    pos_start = find_closest_index(seqs[i][1], pos_start)
    pos_end   = find_closest_index(seqs[i][1], pos_end)
    
    return sequence, neg_start, neg_end, pos_start, pos_end