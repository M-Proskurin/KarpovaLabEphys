import os
import glob
import numpy as np
from scipy import stats

import scipy.io as sio
from utils.import_dio import readTrodesExtractedDataFile

SAMPLE_RATE = 30000.0

def load_DIO_data(ephys_folder, rig_number: int):
    """
    Load DIO data from the specified ephys folder.
    """
    filepath = os.path.join(ephys_folder, "customed_export","DIO.mat")
    if os.path.exists(filepath):
        print("Loading DIO data from .mat file.")
        mat_data = sio.loadmat(filepath)
        mat_to_save = mat_data['mat_to_save']
        DIO = np.floor(np.mod(mat_to_save[:, 1], 2**rig_number) / 2**(rig_number-1))
        ind_state = np.diff(np.concatenate(([0], DIO))) == 1
        TSESync = mat_to_save[ind_state, 0]


    else:
        filepath = glob.glob(os.path.join(ephys_folder, "*DIO", f"*Din{rig_number}.dat"))
        fields = readTrodesExtractedDataFile(filepath[0])
        synch_data = fields['data']
        synch_data = np.array(synch_data)
        time = synch_data['time']
        state = synch_data['state']
        TSESync = time[state==1]

    return TSESync

def find_sync_mapping(TSBSync, TSESync, interval_tolerance=0.02, min_sequence_length=10):
    """
    Find the best linear mapping between two timestamp arrays using inter-pulse intervals.
    
    Parameters:
    TSBSync: array of behavior sync timestamps
    TSESync: array of ephys sync timestamps  
    interval_tolerance: tolerance for matching intervals in seconds (default 20 ms)
    min_sequence_length: minimum length of matching sequences to consider
    min_matches: minimum number of matches required for reliable mapping
    
    Returns:
    dict with 'slope', 'intercept', 'r_squared', 'n_matches', and conversion functions
    """
    
    # Convert to numpy arrays 
    TSB = np.array(TSBSync)
    TSE = np.array(TSESync) / SAMPLE_RATE
    
    # Calculate inter-pulse intervals (differences)
    intervals_B = np.diff(TSB)
    intervals_E = np.diff(TSE)
    
    print(f"Behavior intervals: {len(intervals_B)}, Ephys intervals: {len(intervals_E)}")
    
    def find_matching_sequence(intervals_ref, intervals_target, start_ref, direction=1):
        """
        Find matching sequence starting from start_ref position.
        direction: 1 for forward search, -1 for backward search
        """
        if direction == 1:
            seq_ref = intervals_ref[start_ref:start_ref + min_sequence_length]
        else:
            seq_ref = intervals_ref[start_ref - min_sequence_length + 1:start_ref + 1]
        
        # Search for this sequence in the target array       
        search_range = len(intervals_target) - min_sequence_length + 1
        for start_target in range(search_range):
            seq_target = intervals_target[start_target:start_target + min_sequence_length]
            match = abs(seq_ref - seq_target) <= interval_tolerance
            if sum(match) == min_sequence_length:
                return start_target
        return -1    
    
    # Find matches from beginning
    start_matches = []
    start_b = 0
    while start_b <= len(intervals_B) - min_sequence_length:
        match_start_e = find_matching_sequence(intervals_B, intervals_E, start_b, direction=1)
        if match_start_e != -1:
            # Record timestamp pairs for this sequence
            for i in range(min_sequence_length):
                start_matches.append((TSE[match_start_e + i + 1], TSB[start_b + i + 1]))
            # print(f"Found start sequence match: B[{start_b}] -> E[{match_start_e}]")
            break
        start_b += 1
    
    # Find matches from end
    end_matches = []
    start_b = len(intervals_B) - 1
    while start_b >= min_sequence_length - 1:
        match_start_e = find_matching_sequence(intervals_B, intervals_E, start_b, direction=-1)
        if match_start_e != -1:
            # Record timestamp pairs for this sequence (adjust for backward search)
            match_end_e = match_start_e + min_sequence_length - 1
            for i in range(min_sequence_length):
                end_matches.append((TSE[match_end_e - i], TSB[start_b - i]))
            # print(f"Found end sequence match: B[{start_b}] -> E[{match_end_e}]")
            break
        start_b -= 1
    
    # Combine all matches
    all_matches = start_matches + end_matches
    
    # Extract matched timestamps
    matched_ephys = np.array([match[0] for match in all_matches])
    matched_behavior = np.array([match[1] for match in all_matches])
    
    # Remove duplicates if any
    unique_pairs = list(set(zip(matched_ephys, matched_behavior)))
    matched_ephys = np.array([pair[0] for pair in unique_pairs])
    matched_behavior = np.array([pair[1] for pair in unique_pairs])

    # print(f"Found {len(matched_ephys)} unique matching timestamp pairs")

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(matched_ephys, matched_behavior)
    
    # Create mapping dictionary
    best_mapping = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'n_matches': len(matched_ephys),
        'p_value': p_value,
        'std_err': std_err,
        'matched_ephys': matched_ephys,
        'matched_behavior': matched_behavior
    }
    print("Interpolation slope", best_mapping["slope"])

    # Add conversion functions
    def ephys_to_behavior(t_ephys):
        """Convert ephys timestamps to behavior timestamps"""
        return best_mapping['slope'] * t_ephys / SAMPLE_RATE + best_mapping['intercept']
    
    def behavior_to_ephys(t_behavior):
        """Convert behavior timestamps to ephys timestamps"""
        return (t_behavior - best_mapping['intercept']) / best_mapping['slope'] * SAMPLE_RATE

    best_mapping['ephys_to_behavior'] = ephys_to_behavior
    best_mapping['behavior_to_ephys'] = behavior_to_ephys
    
    return best_mapping


