# openclose fists 
# session 1 plot labels and view distribtuion (time series, fft), rest and openclose fists
# session 2 plot labels and view distribution (time series, fft), rest and openclose fists
import os
import re
from datetime import datetime
import utils
import numpy as np
import matplotlib.pyplot as plt
from mne import Epochs, find_events, create_info
from mne.io import RawArray

def load_oc_data(data_folder:str):
    """
    Load the open-close fist/feet data, one subject only.
    """
    def extract_dt(path):
        # sub-EB-43_EEG_recording_2025-06-08-19.57.52.csv
        ts = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2})", path).group(1)
        return datetime.strptime(ts, "%Y-%m-%d-%H.%M.%S")
    eeg_paths = []
    for fname in os.listdir(data_folder):
        if "EEG" in fname and ".csv" in fname:
            eeg_paths.append(os.path.join(data_folder, fname))
    eeg_paths.sort(key=extract_dt)
    exp1, exp2 = [], []
    for idx, fp in enumerate(eeg_paths): #0,1,2,3
        (exp1 if idx % 2 == 0 else exp2).append([fp]) #even number is for open/close fists

    print("Experiment-1 (fists) files:", exp1) #open/close fists
    print("Experiment-2 (feet)  files:", exp2) #open/close feet
    return exp1, exp2


def loadCSVFile(csvFilename, exp_num, 
                      sfreq=256, channels=("TP9", "AF7", "AF8", "TP10"),
                      stim_ind=5,
                      t_min=0.0,
                      t_max=4.0,
                      l_freq=1,
                      h_freq=30,
                      do_filter=False):
    #predit the csvFilanme giving model
    CODEMAP = {
    1: {"target": 101, "non-target": 201},   # exp 1: open-close fist
    2: {"target": 102, "non-target": 202},   # exp 2: open-close feet
}
    raw = utils.load_muse_csv_as_raw(
        csvFilename,
        sfreq=sfreq,
        ch_ind=[0, 1, 2, 3],    
        stim_ind=stim_ind,     
        replace_ch_names=None
    )
    raw.load_data()
    if do_filter:
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir')

    code = CODEMAP[exp_num]
    events = find_events(raw, shortest_event=1)
    event_id = {"Target": code["target"],
                "Non-Target": code["non-target"]}
    epochs = Epochs(raw, events, event_id,
                    tmin=t_min, tmax=t_max,
                    picks=list(channels),
                    baseline=None, preload=True)
    
    X = epochs.get_data() *1e6                                  # (n_ev, 4, 1025)
    X = X[:, :, :-1]  
    print("X shape:", X.shape)
    y = (epochs.events[:, 2] == event_id['Target']).astype(int)
    return X, y 

folder_1_path = r"C:\Users\luoyu\Downloads\Muse_data_OpenCloseFistsFeet_session1"
folder_2_path = r"C:\Users\luoyu\Downloads\Muse_data_OpenCloseFists_session2"

exp1_s1, exp2_s1 = load_oc_data(folder_1_path)


# Collect EEG CSV files and sort them by timestamp
import re
from datetime import datetime

def extract_timestamp(filename):
    # Extract the timestamp part using regex
    match = re.search(r'_recording_(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2})\.csv', filename)
    if match:
        timestamp_str = match.group(1)
        # Convert to datetime object for proper comparison
        timestamp_str = timestamp_str.replace('.', ':')
        return datetime.strptime(timestamp_str, '%Y-%m-%d-%H:%M:%S')
    return datetime.min  # Fallback value

# Collect and sort files
files = [f for f in os.listdir(folder_2_path) if "EEG" in f and f.endswith(".csv")]
sorted_files = sorted(files, key=extract_timestamp)

# Create exp1_s2 with sorted files
exp1_s2 = []
for file in sorted_files:
    exp1_s2.append([os.path.join(folder_2_path, file)])

# Initialize data structure with both fist and feet
data = {
    'fist': {'session_1': {}, 'session_2': {}},
    'feet': {'session_1': {}}
}

# Process fist data for session 1
for i in range(len(exp1_s1)):
    run = "run_" + str(i + 1)
    
    # Load the file for this experiment and run
    file = exp1_s1[i][0]
    
    X, y = loadCSVFile([file], 1)
    
    # Store the data
    data['fist']['session_1'][run] = {
        'X': X,
        'y': y,
        'file_path': exp1_s1[i]
    }
    
# Process feet data for session 1
for i in range(len(exp2_s1)):
    run = "run_" + str(i + 1)
    
    # Load the file for this experiment and run
    file = exp2_s1[i][0]
    
    X, y = loadCSVFile([file], 2)
    
    # Store the data
    data['feet']['session_1'][run] = {
        'X': X,
        'y': y,
        'file_path': exp2_s1[i]
    }
    
for i in range(len(exp1_s2)):
    run = "run_" + str(i + 1)
    
    # Load the file for this experiment and run
    file = exp1_s2[i][0]
    
    X, y = loadCSVFile([file], 1)
    data['fist']['session_2'][run] = {
        'X': X,
        'y': y,
        'file_path': exp1_s2[i]
    }

# Define a function to plot data for each run with channels as subplots
def plot_channels_by_target(data, exp_type, session_key, run_key, channels=['TP9', 'AF7', 'AF8', 'TP10']):
    run_data = data[exp_type][session_key][run_key]
    X = run_data['X']
    y = run_data['y']
    file_path = run_data['file_path']
    
    # Create time axis (assuming 256 Hz sampling rate)
    time = np.arange(X.shape[2]) / 256.0  # Convert to seconds
    
    # Create figure with 4 subplots (one for each channel)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    # Get indices for different targets
    target_0_idx = np.where(y == 0)[0]
    target_1_idx = np.where(y == 1)[0]
    
    # Set colors
    colors = ['blue', 'red']
    
    # Plot each channel
    for ch_idx in range(4):
        # Target 0 data
        if len(target_0_idx) > 0:
            target_0_data = X[target_0_idx, ch_idx, :]
            mean_0 = np.mean(target_0_data, axis=0)
            std_0 = np.std(target_0_data, axis=0)
            axs[ch_idx].plot(time, mean_0, color=colors[0], label='Target 0 (Resting)')
            axs[ch_idx].fill_between(time, mean_0-std_0, mean_0+std_0, color=colors[0], alpha=0.2)
        
        # Target 1 data
        if len(target_1_idx) > 0:
            target_1_data = X[target_1_idx, ch_idx, :]
            mean_1 = np.mean(target_1_data, axis=0)
            std_1 = np.std(target_1_data, axis=0)
            if exp_type == 'fist':
                target_label = 'Target 1 (OpenCloseFist)'
            else:  # feet
                target_label = 'Target 1 (OpenCloseFeet)'
            axs[ch_idx].plot(time, mean_1, color=colors[1], label=target_label)
            axs[ch_idx].fill_between(time, mean_1-std_1, mean_1+std_1, color=colors[1], alpha=0.2)
        
        axs[ch_idx].set_title(f'Channel: {channels[ch_idx]}', fontsize=30)
        axs[ch_idx].set_xlabel('Time (s)', fontsize=30)
        axs[ch_idx].set_ylabel('Amplitude (\u03bcV)', fontsize=30)
        axs[ch_idx].set_ylim(-200, 100)  # Set y-axis limits from -200 to 100 microvolts
        axs[ch_idx].legend(loc='lower right', fontsize=22)
        axs[ch_idx].grid(True)
    
    # Add overall title with sample count information
    # total_samples = len(y)
    # target_0_count = len(target_0_idx)
    # target_1_count = len(target_1_idx)
    #fig.suptitle(f'EEG Channels for {exp_type.capitalize()}: {session_key}, {run_key}\nTotal: {total_samples} samples (Target 0: {target_0_count}, Target 1: {target_1_count})', fontsize=14)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    plt.savefig(f'{output_dir}/{exp_type}_{session_key}_{run_key}_channel_plot.png', dpi=600, bbox_inches='tight')
    plt.close()

# Already imported matplotlib at the top

# Print the keys available in the data structure
print("\nFIST DATA:")
if 'fist' in data:
    print("Available sessions:", list(data['fist'].keys()))
    print("Runs in session_1:", list(data['fist']['session_1'].keys()))
    if 'session_2' in data['fist']:
        print("Runs in session_2:", list(data['fist']['session_2'].keys()))
    else:
        print("No session_2 data found")
else:
    print("No fist data loaded")
    
print("\nFEET DATA:")
if 'feet' in data:
    print("Available sessions:", list(data['feet'].keys()))
    print("Runs in session_1:", list(data['feet']['session_1'].keys()))
else:
    print("No feet data loaded")

print("\nGenerating plots for each run...")

# Plot fist data
print("\nPlotting FIST data:")
if 'fist' in data:
    # Plot each run in session_1
    for run_key in data['fist']['session_1']:
        print(f"Plotting fist {run_key}...")
        plot_channels_by_target(data, 'fist', 'session_1', run_key)
    
    # Plot session_2 if it exists
    if 'session_2' in data['fist'] and data['fist']['session_2']:
        for run_key in data['fist']['session_2']:
            print(f"Plotting fist {run_key}...")
            plot_channels_by_target(data, 'fist', 'session_2', run_key)

# Plot feet data
print("\nPlotting FEET data:")
if 'feet' in data:
    # Plot each run in session_1
    for run_key in data['feet']['session_1']:
        print(f"Plotting feet {run_key}...")
        plot_channels_by_target(data, 'feet', 'session_1', run_key)
    
print("All plots saved to 'plots' directory")

# Function to plot raw EEG data using MNE
def plot_raw_eeg_data(file_path, duration=16.0, start_time=0.0, channels=['TP9', 'AF7', 'AF8', 'TP10'], save_plot=True):
    """
    Plot raw EEG data using MNE's built-in plotting function.
    
    Parameters:
    ----------
    file_path : str
        Path to the EEG data CSV file
    duration : float, default=16.0
        Duration of data to plot in seconds
    start_time : float, default=0.0
        Start time in seconds
    channels : list, default=['TP9', 'AF7', 'AF8', 'TP10']
        List of channel names to plot
    """
    # Load the raw data
    raw = utils.load_muse_csv_as_raw(
        [file_path],
        sfreq=256,
        ch_ind=[0, 1, 2, 3],
        stim_ind=5,
        replace_ch_names=None
    )
    
    # Scale data to microvolts for better visualization
    raw._data *= 1e6
    
    # Set channel types to EEG
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch != 'Stim'})
    
    # Plot the raw data
    fig = raw.plot(duration=duration, start=start_time, scalings='auto', 
                    title=f'Raw EEG Data - {os.path.basename(file_path)}',
                    show_scrollbars=True, show=True)
    
    # Save the figure if requested
    if save_plot:
        # Create output directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Create a meaningful filename based on the input parameters
        output_path = f'plots/raw_eeg_start{int(start_time)}s_dur{int(duration)}s.png'
        
        # Create a single plot for all channels with specified colors
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        
        # Remove the frame/border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)  # Hide left border too
        
        # Define softer colors for each channel
        channel_colors = {
            'TP9': '#7fb3d5',    # Muted blue
            'AF7': '#f1948a',    # Muted red
            'AF8': '#7dcea0',    # Muted green
            'TP10': '#bb8fce'   # Muted purple
        }
        
        # Track the offset value for each channel to visually separate them
        offsets = {}
        offset_values = [600, 200, -200, -600]  # Offset values to visually separate channels
        
        # Create relative time array starting from 0
        time_segment = int(duration * raw.info['sfreq'])  # Number of samples for the duration
        start_idx = int(start_time * raw.info['sfreq'])  # Starting index
        relative_times = np.linspace(0, duration, time_segment)
        
        # Plot each channel with different colors and offsets
        for i, ch_name in enumerate(raw.ch_names):
            if ch_name == 'Stim':
                continue
                
            # Get data for this channel and the time range
            data, times = raw[i, start_idx:start_idx + time_segment]
            
            # Apply offset to separate channels visually
            offsets[ch_name] = offset_values[i]
            data_offset = data.T + offsets[ch_name]
            
            # Plot with specified color
            ax.plot(relative_times, data_offset, color=channel_colors[ch_name], linewidth=1)
            
            # Add a solid gray horizontal line to indicate the baseline for this channel
            ax.axhline(y=offsets[ch_name], color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add channel label directly next to the line on the left side
            ax.text(-0.5, offsets[ch_name], ch_name, color='black', 
                   fontsize=16, va='center', ha='right', fontweight='bold')
        
        # Add labels but no grid
        ax.grid(False)
        ax.set_xlabel('Time (s)', fontsize=14)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        
        # Set x-axis ticks to show time from 0 to 16
        ax.set_xticks(np.arange(0, duration+1, 2))
        ax.set_xticklabels(np.arange(0, duration+1, 2), fontsize=12)
        
        # Set main title with larger font
        plt.title('Raw EEG Data from Muse LSL', fontsize=18)
        
        # Adjust spacing
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Raw EEG plot saved to {output_path}")
    
    # Return the original interactive figure
    return fig

# Plot multiple segments of raw EEG data from session 1 run 2
print("\nGenerating raw EEG plots for session 1 run 2...")
file_path = exp1_s1[1][0]  # Session 1 run 2 file path

# Define multiple starting points to generate different plots
start_times = [50, 60, 70, 80]
duration = 16

for start_time in start_times:
    print(f"Generating plot starting at {start_time}s...")
    plot_raw_eeg_data(file_path, duration=16.0, start_time=float(start_time))