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

        if not spike_times_path.exists() or not spike_clusters_path.exists():
            raise FileNotFoundError("Spike times or clusters file not found in the specified directory.")

        self.spike_times = np.load(spike_times_path)
        self.spike_clusters = np.load(spike_clusters_path)

        if cluster_info_path.exists():
            self.cluster_info = pd.read_csv(cluster_info_path,sep = '\t')
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

    
    def get_spike_data(self):
        """Return spike times and cluster assignments."""
        if self.spike_times is None or self.spike_clusters is None:
            raise ValueError("Spike data not loaded. Call load_spike_data() first.")
        
        return self.spike_times, self.spike_clusters, self.cluster_info