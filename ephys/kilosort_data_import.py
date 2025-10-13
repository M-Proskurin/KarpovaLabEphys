import os
from pathlib import Path

import numpy as np
import pandas as pd

class KilosortData:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.cluster_info = None
        self.locate_KS_folder()
        self.load_spike_data()
        self.select_clusters()
        self.extract_cluster_properties()
        

    def locate_KS_folder(self):
        # locate Kilosort output folder: look for first subfolder matching '*kilosort*'
        try:
            kilosort_parents = [p for p in self.data_dir.iterdir() if p.is_dir() and "kilosort" in p.name.lower()]
        except Exception:
            kilosort_parents = []

        if kilosort_parents:
            ks_parent = kilosort_parents[0]
            ks_candidate = ks_parent / "Kilosort4"
            self.KSfolder = ks_candidate if ks_candidate.exists() else None
        else:
            self.KSfolder = None

    def load_spike_data(self):
        """Load spike times and cluster assignments from Kilosort output files."""
        spike_times_path = self.KSfolder / 'spike_times.npy'
        spike_clusters_path = self.KSfolder / 'spike_clusters.npy'
        cluster_info_path = self.KSfolder / 'cluster_info.tsv'
        channel_map_path = self.KSfolder / 'channel_map.npy'

        if not spike_times_path.exists() or not spike_clusters_path.exists():
            raise FileNotFoundError("Spike times or clusters file not found in the specified directory.")

        self.spike_times = np.load(spike_times_path)
        self.spike_clusters = np.load(spike_clusters_path)
        self.channel_map = np.load(channel_map_path) if channel_map_path.exists() else None

        if cluster_info_path.exists():
            cluster_info = pd.read_csv(cluster_info_path, sep='\t')
            cluster_info = cluster_info.sort_values(by=["depth","cluster_id"], ascending=[False, True]).reset_index(drop=True)
            self.cluster_info = cluster_info
        else:
            print("Cluster info file not found. Using KS labels.")
            self.cluster_info = None
        self.ks_labels = pd.read_csv(self.KSfolder / 'cluster_KSLabel.tsv',sep = '\t')

    def select_clusters(self):
        """Select specific clusters to load based on provided cluster IDs."""
        if self.cluster_info is None:
            ci = self.ks_labels
        else:
            ci = self.cluster_info
        mask = pd.Series(False, index=ci.index)
        if "KSLabel" in ci.columns:
            mask = mask | (ci["KSLabel"].astype(str).str.lower() == "good")
        if "group" in ci.columns:
            mask = mask | (ci["group"].astype(str).str.lower() == "good")
            ci2 = ci.loc[mask, ["group"]]
            mask2 = (ci2["group"].astype(str).str.lower() == "good")
            self.curated_cells = mask2.to_numpy(dtype=bool)
        # store as numpy boolean array for downstream code expecting array
        self.to_load = mask.to_numpy(dtype=bool)

    def extract_cluster_properties(self):
        """Extract properties like channel, amplitude, firing rate for selected clusters."""
        channel_map = self.channel_map
        to_load = self.to_load
        if self.cluster_info is None:
            print("The session is not curated! Using KS labels.")
            ci = self.ks_labels
            ks_ids = ci["cluster_id"].tolist()
            ks_ids = [ks_ids[i] for i, load in enumerate(to_load) if load]
            channel = self.waveform2channel()
            channel = np.array([channel[i] for i, load in enumerate(to_load) if load])
            ks_channel = channel_map[channel]
            amplitude_df = pd.read_csv(self.KSfolder / 'cluster_Amplitude.tsv', sep='\t') 
            amplitude = amplitude_df.loc[amplitude_df["cluster_id"].isin(ks_ids), ["Amplitude"]]
            fr = np.empty(len(ks_ids))
            amp = np.empty(len(ks_ids))
        else:
            ci = self.cluster_info
            ks_ids = ci["cluster_id"].tolist()
            ks_ids = [ks_ids[i] for i, load in enumerate(to_load) if load]
            ks_channel = ci.loc[ci["cluster_id"].isin(ks_ids), ["ch"]]
            amplitude = ci.loc[ci["cluster_id"].isin(ks_ids), ["Amplitude"]]
            amp = ci.loc[ci["cluster_id"].isin(ks_ids), ["amp"]]
            fr = ci.loc[ci["cluster_id"].isin(ks_ids), ["fr"]]
            channel_map = np.array(channel_map)   # ensure numpy array
            ks_channel = np.array(ks_channel)
            channel = np.array([np.where(channel_map == a)[0][0] for a in ks_channel])

        channel_positions = np.load(self.KSfolder / 'channel_positions.npy') 
        DV = channel_positions[tuple(channel),1]
        XX = channel_positions[tuple(channel),0]

        self.channel = ks_channel.squeeze()
        self.amplitude = np.array(amplitude).squeeze()
        self.fr = np.array(fr).squeeze()
        self.amp = np.array(amp).squeeze()
        self.ks_ids = ks_ids
        self.DV = DV
        self.XX = XX

    def waveform2channel(self):
        """
        Find the channel with the largest peak-to-peak amplitude for each template.
        """
        p = Path(self.KSfolder)
        templates_path = p / "templates.npy"
        templates = np.load(templates_path) 

        n_templates = templates.shape[0]
        channels = np.empty(n_templates, dtype=int)

        for i in range(n_templates):
            T = templates[i]  # 2D array for this template
            ptp_per_channel = T.max(axis=0) - T.min(axis=0)
            # pick channel with largest peak-to-peak; argmax returns first on ties
            ch_idx = int(np.argmax(ptp_per_channel))
            channels[i] = ch_idx

        return channels

    def get_spike_data(self):
        """Return spike times and cluster assignments."""
        if self.spike_times is None or self.spike_clusters is None:
            raise ValueError("Spike data not loaded. Call load_spike_data() first.")
        
        return self.spike_times, self.spike_clusters, self.cluster_info