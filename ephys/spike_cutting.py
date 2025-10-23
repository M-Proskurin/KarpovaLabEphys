import numpy as np
from scipy import stats
import warnings


def bin_spikes(all_spike_times, spike_bin_size):
    """
    Bin spike times into time bins.
    
    Parameters:
    all_spike_times: list of arrays, each containing spike times for one unit
    spike_bin_size: bin size in seconds
    
    Returns:
    last_spike_time: maximum spike time across all units
    spike_matrix: list of binned spike counts for each unit
    """
    # Find the maximum spike time across all units
    last_spike_time = 0
    last_spike_time = max(np.max(spikes) for spikes in all_spike_times if len(spikes) > 0)
    last_spike_time = np.ceil(last_spike_time / spike_bin_size) * spike_bin_size
    
    # Create time bins
    bin_edges = np.arange(0, last_spike_time + spike_bin_size, spike_bin_size)
    n_bins = len(bin_edges) - 1
    n_neurons = len(all_spike_times)

    # Create spike matrix ( neurons x time_bins )
    spike_matrix = np.zeros((n_neurons, n_bins), dtype=int)

    for i, spikes in enumerate(all_spike_times):
        if len(spikes) > 0:
            counts, _ = np.histogram(spikes, bins=bin_edges)
            spike_matrix[i, :] = counts

    return last_spike_time, spike_matrix


def anscombe_transform(spike_matrix):
    """
    Apply Anscombe transformation to stabilize variance.
    For Poisson data: 2 * sqrt(x + 3/8)
    """
    return 2 * np.sqrt(spike_matrix + 3/8)


def convolve_matrix(spike_matrix, bins_to_average):
    """
    Smooth spike matrix by averaging over bins.
    
    Parameters:
    spike_matrix: 2D array (time_bins x neurons)
    bins_to_average: number of bins to average
    
    Returns:
    smoothed matrix
    """

    import numpy as np

    # Create uniform kernel (normalized so total weight = 1)
    uniform_kernel = np.ones(bins_to_average, dtype=float) / bins_to_average

    # Convolve each row (neuron) independently, same output length
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, uniform_kernel, mode='same'),
        axis=1,
        arr=spike_matrix
    )

    return smoothed


def spike_cutting(b, e, spike_bin_size=0.1, anscombe_true=False, bins_to_average=1, 
                 time_around_poke=1, **kwargs):
    """
    Cut spike data around behavioral events.
    
    Parameters:
    b: behavior object with timesEphys attribute
    e: ephys object with allSpikeTimes attribute
    spike_bin_size: bin size for spike trains (seconds)
    anscombe_true: whether to apply Anscombe transformation
    bins_to_average: number of bins to average for smoothing
    time_around_poke: time window around events (seconds)
    
    Returns:
    dict with windows, warnings, and other analysis results
    """
    
    # Handle additional keyword arguments
    for key, value in kwargs.items():
        if key == 'spike_bin_size':
            spike_bin_size = value
        elif key == 'anscombe_true':
            anscombe_true = value
        elif key == 'bins_to_average':
            bins_to_average = value
        elif key == 'time_around_poke':
            time_around_poke = value
    
    # Get behavior times
    times = b.times  # This should be a list of arrays with event times
    
    # Bin spikes
    last_spike_time, spike_matrix = bin_spikes(e.all_spike_times, spike_bin_size)
    
    # Apply transformations
    if anscombe_true:
        spike_matrix = anscombe_transform(spike_matrix)
    
    if bins_to_average > 1:
        spike_matrix = convolve_matrix(spike_matrix, bins_to_average)
    
    # Get dimensions
    n_bins = spike_matrix.shape[1]  # number of time bins
    n_neurons = spike_matrix.shape[0]  # number of neurons
    
    # Convert event times to bin indices
    times_bins = np.round(times / last_spike_time * n_bins).astype(int)
    n_events = times.shape[0]

    # Define time window around events
    window_bins = np.arange(-time_around_poke/spike_bin_size, 
                  time_around_poke/spike_bin_size + 1, dtype=int)
    window_length = len(window_bins)
  
    print(f"Processing {len(times)} time anchor types and {n_neurons} neurons...")
    
    windows = []  # Initialize output structures
    # Main loop - extract windows around events
    for ti in range(times.shape[1]):  # select time anchors
        w2 = []  # windows for this time type
        for ni in range(n_neurons):  # select neurons
            w = np.zeros((n_events, window_length))
            for fi in range(n_events):  # for each event
                # Extract window, handling boundaries
                indices = times_bins[fi][ti] + window_bins
                indices = np.clip(indices, 0, n_bins-1).astype(int)  # clip to valid range
                w[fi, :] = spike_matrix[ni, indices]
            w2.append(w)
        windows.append(w2)

    # Return results
    results = {
        'windows': windows,
        'spike_matrix': spike_matrix,
        'last_spike_time': last_spike_time,
        'parameters': {
            'spike_bin_size': spike_bin_size,
            'anscombe_true': anscombe_true,
            'bins_to_average': bins_to_average,
            'time_around_poke': time_around_poke
        }
    }
    
    return results


# Example usage function
def plot_spike_cutting_results(results, time_idx=0, neuron_idx=0):
    """
    Plot results from spike cutting analysis.
    """
    import matplotlib.pyplot as plt
    
    windows = results['windows']
    x = results['x']
    spike_bin_size = results['parameters']['spike_bin_size']
    
    if time_idx in windows and neuron_idx in windows[time_idx]:
        data = windows[time_idx][neuron_idx]
        
        plt.figure(figsize=(12, 6))
        
        # Plot individual trials
        plt.subplot(1, 2, 1)
        plt.imshow(data, aspect='auto', cmap='viridis', 
                  extent=[x[0]*spike_bin_size, x[-1]*spike_bin_size, 0, data.shape[0]])
        plt.xlabel('Time relative to event (s)')
        plt.ylabel('Trial')
        plt.title(f'Spike Raster - Time Type {time_idx}, Neuron {neuron_idx}')
        plt.colorbar(label='Spike Count')
        
        # Plot average response
        plt.subplot(1, 2, 2)
        mean_response = np.mean(data, axis=0)
        sem_response = np.std(data, axis=0) / np.sqrt(data.shape[0])
        time_axis = x * spike_bin_size
        
        plt.plot(time_axis, mean_response, 'b-', linewidth=2)
        plt.fill_between(time_axis, 
                        mean_response - sem_response,
                        mean_response + sem_response,
                        alpha=0.3, color='blue')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Time relative to event (s)')
        plt.ylabel('Average Spike Count')
        plt.title(f'Average Response - Time Type {time_idx}, Neuron {neuron_idx}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data found for time_idx={time_idx}, neuron_idx={neuron_idx}")


if __name__ == "__main__":
    # Example usage would go here
    pass