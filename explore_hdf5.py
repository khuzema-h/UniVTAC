import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, "Dataset", obj.shape, obj.dtype)
    elif isinstance(obj, h5py.Group):
        print(name, "Group")

file_path = '/home/khuzema/UniVTAC/eval_result/ACT/insert_HDMI/deploy_policy_insert_HDMI/2026-04-16_15:03:40/hdf5/1000000.hdf5'
with h5py.File(file_path, 'r') as f:
    f.visititems(print_structure)
