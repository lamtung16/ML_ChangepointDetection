# DOCSTRING
"""
The program applies the LOPART algorithm to analyze toy data, 
indicating a focus on testing or demonstrating the algorithm's functionality in a controlled setting.
"""


# IMPORT
import numpy as np
import matplotlib.pyplot as plt


# FUNCTIONS
# generate data: sequence, labels
def gen_data():
    np.random.seed(123)

    # Generate a sequence with 100 numbers
    sequence_length = 100
    sequence = np.zeros(sequence_length)

    means = [50, 30, 60, 40]                # Define the means for the 4 segments
    segment_lengths = [25, 25, 25, 25]      # Define the lengths of the segments

    # Populate the sequence with segments having different means
    start_index = 0
    for mean, length in zip(means, segment_lengths):
        end_index = start_index + length
        sequence[start_index:end_index] = np.random.normal(loc=mean, scale=5, size=length)
        start_index = end_index

    # outlier
    sequence[62] = 30

    # Labels
    neg_start = [5, 55]
    neg_end   = [15, 65]
    pos_start = [20, 70]
    pos_end   = [30, 80]

    return sequence, neg_start, neg_end, pos_start, pos_end


# function to plot sequence with labels and changepoints wrt lambda (if provided)
def plot_sequence(sequence, neg_start, neg_end, pos_start, pos_end, mean=None, lda=None):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the figure title
    if(lda is not None):
        fig.suptitle('lambda = ' + str(lda))

    # Plot the regions
    bounds = (np.min(sequence) - 10, np.max(sequence) + 10)
    for start, end in zip(neg_start, neg_end):
        ax.fill_betweenx(bounds, start-1, end-1, color='pink', alpha=0.5, label='negative region')

    for start, end in zip(pos_start, pos_end):
        ax.fill_betweenx(bounds, start-1, end-1, color='red', alpha=0.5, label='positive region')

    # Plot the sequence
    ax.plot(sequence, marker='o', color='blue', label='sequence')

    # Plot the mean
    if(mean is not None):
        ax.plot(mean, color='green', label='mean')

    # Set axis labels
    ax.set_xlabel('point')
    ax.set_ylabel('value')

    # Set limit
    ax.set_xlim(-5, len(sequence) + 5)
    ax.set_ylim(bounds)

    # show legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels)

    # Show the plot
    plt.show()


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
        tau = tau_star[tau]
    return np.append(0, chpnt)


# function to get the mean vector of the sequence from the list of changepoint
def get_mean(sequence, chpnt):
    mean = np.zeros(len(sequence))
    for i in range(len(chpnt)-1):
        mean[chpnt[i]:chpnt[i+1]] = np.mean(sequence[chpnt[i]:chpnt[i+1]])
    return mean


# function to get T - set of possible changepoint wrt each position
def get_T(sequence_length, neg_start, neg_end, pos_start, pos_end):
    T = []
    T.append([])

    for i in range(1, sequence_length+1):
        for j in range(len(neg_start)):
            if ((neg_start[j] < i and i <= neg_end[j]) or (pos_start[j] < i and i < pos_end[j])):   # i inside regions
                T.append(T[i-1])
                break
            elif (i == pos_end[j]):                                                                 # i is just outside positive region
                region = []
                for k in range(pos_start[j], pos_end[j]):
                    region.append(k)
                T.append(region)
                break

        if (len(T) == i):                                                                           # otherwise
            T.append(T[i-1]+[i-1])
    
    return T


# lopart dynamic algorithm return set of changepoints given lambda, T (set of possible changepoints), sequence, and cumsum vectors
def lopart(lda, T, sequence, y, z):
    sequence_length = len(sequence)

    # Set up
    C = np.zeros(sequence_length)
    C[0] = -lda

    # Get tau_star
    tau_star = np.zeros(sequence_length, dtype=int)
    for i in range(1, sequence_length):

        # get set of possible changepoint
        po_chpnt = T[i]

        # get set of possible value
        V = np.inf * np.ones(sequence_length)
        for j in po_chpnt:
            V[j] = C[j] + lda + L(j+1, i, y, z)

        # get optimal tau from set V
        last_chpnt = np.argmin(V)

        # update C_i
        C[i] = V[last_chpnt]

        # update tau_star
        tau_star[i] = last_chpnt

    # get set of changepoints
    set_of_chpnt = trace_back(tau_star)
    
    return set_of_chpnt


# Main block
if __name__ == "__main__":
    # generate data
    sequence, neg_start, neg_end, pos_start, pos_end = gen_data()
    sequence_length = len(sequence)
    
    # plot data
    plot_sequence(sequence, neg_start, neg_end, pos_start, pos_end)

    # vectors of cumulative sums
    y = np.zeros(sequence_length + 1)       # vectors of cumulative sums of data
    z = np.zeros(sequence_length + 1)       # vectors of cumulative sums of squares

    for i in range(1, sequence_length+1):
        y[i] = y[i-1] + sequence[i-1]
        z[i] = z[i-1] + np.square(sequence[i-1])
    
    # get T
    T = get_T(sequence_length, neg_start, neg_end, pos_start, pos_end)

    # plot solution with respect to different lambda
    for lda in [0, 1, 10, 100, 1000, 10000, 100000]:
        chpnt = lopart(lda, T, sequence, y, z)
        mean  = get_mean(sequence, chpnt)
        plot_sequence(sequence, neg_start, neg_end, pos_start, pos_end, mean, lda)