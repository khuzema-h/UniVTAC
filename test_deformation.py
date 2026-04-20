import h5py
import numpy as np
f = h5py.File('/home/khuzema/UniVTAC/eval_result/ACT/insert_HDMI/deploy_policy_insert_HDMI/2026-04-16_15:03:40/hdf5/1000000.hdf5', 'r')
ee = f['embodiment']['ee']
left = f['tactile']['left_tactile']['marker']
right = f['tactile']['right_tactile']['marker']

for i in range(100):
    start = max(0, i-4)
    v_left = left[i] - left[start]
    v_left_mean = np.mean(v_left, axis=0)
    def_left = np.mean(np.linalg.norm(v_left - v_left_mean, axis=-1))
    
    v_right = right[i] - right[start]
    v_right_mean = np.mean(v_right, axis=0)
    def_right = np.mean(np.linalg.norm(v_right - v_right_mean, axis=-1))
    
    deformation = max(def_left, def_right)
    
    z_vel = ee[i][2] - ee[start][2]
    
    print(f"Frame {i:03d} | Z-vel: {z_vel:.4f} | Deformation: {deformation:.4f}")
