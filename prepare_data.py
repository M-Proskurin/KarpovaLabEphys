from utils.file_utils import search_files
import behavior.PyControl_data_import as bimp
import behavior.parse_pycontrol as ex
import ephys.ephys2behavior as e2b
import ephys.kilosort_data_import as eimp
import ephys.spike_cutting as sc


behavior_file, ephys_folder = search_files("662", "06122024")
print("Behavior file:", behavior_file)
print("Ephys folder:", ephys_folder)

session = bimp.Session(behavior_file)
extractor = ex.Extractor(session)
b = ex.Behavior(extractor)


TSBSync = b.TSBSync
TSESync = e2b.load_DIO_data(ephys_folder, 4)
sync_mapping = e2b.find_sync_mapping(TSBSync, TSESync)
b.e2b_mapping = sync_mapping['ephys_to_behavior']

e = eimp.KilosortData(ephys_folder)
e.all_spike_times = [b.e2b_mapping(spikes) for spikes in e.allSpikeSI]
e.buggy_periods["buggy_time"] = b.e2b_mapping(e.buggy_periods["buggy_SI"])
spike_data = sc.spike_cutting(
    times=b.times,
    all_spike_times=e.all_spike_times,
    spike_bin_size=0.1,
    anscombe_true=False,
    bins_to_average=1,
    time_around_poke=1
)
e.windows = spike_data['windows']
e.spike_matrix = spike_data['spike_matrix']
e.window_params = spike_data['parameters']