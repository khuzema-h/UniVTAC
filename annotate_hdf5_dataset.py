import h5py
import numpy as np
import ollama
import os
import glob

# Using the larger ~10GB text model available on your system for nuanced decisions
RESOLVER_MODEL = 'gemma4:e4b'

def evaluate_correct_vs_insert(ee_hist, left_marker_hist, right_marker_hist):
    if len(ee_hist) <= 1:
        return b"Insert", b"Insufficient history to evaluate."
        
    z_vel = ee_hist[-1][2] - ee_hist[0][2]
    # Calculate sideways and rotational movement to detect "struggling/adjusting"
    xy_vel = np.linalg.norm(ee_hist[-1][:2] - ee_hist[0][:2])
    rot_vel = np.linalg.norm(ee_hist[-1][3:] - ee_hist[0][3:])
    
    left_disp = np.mean(np.linalg.norm(left_marker_hist[-1] - left_marker_hist[0], axis=-1))
    right_disp = np.mean(np.linalg.norm(right_marker_hist[-1] - right_marker_hist[0], axis=-1))
    mean_marker_disp = max(left_disp, right_disp)
    
    prompt = f"""
The robot has ALREADY reached the insertion phase and is now toggling between 'Insert' and 'Correct'.
Movement data over the last 5 steps:
- Z-axis velocity: {z_vel:.5f} (Negative = downwards into the port)
- XY-axis translation: {xy_vel:.5f} (Moving sideways to adjust)
- Rotational adjustment: {rot_vel:.5f} (Twisting to align)
- Mean Marker Displacement: {mean_marker_disp:.4f} (Contact disturbance)

RULES:
- 'Insert': The robot is actively pushing downwards (Z-vel is negative). XY and Rotational adjustments are minimal.
- 'Correct': The robot is struggling to insert and is adjusting its position. This is indicated by moving upwards/sideways (Z-vel is zero or positive), or having noticeable XY translation or Rotational adjustment to fix the alignment.

Based on this, what is the current phase? 
First, briefly explain your reasoning in 1-2 sentences. 
Then, on a new line, provide your final answer as EXACTLY ONE word: [Insert, Correct].
"""
    try:
        res = ollama.chat(model=RESOLVER_MODEL, messages=[{'role': 'user', 'content': prompt}])
        text = res['message']['content'].strip()
        
        # Check if the final word contains the phase
        last_line = text.split('\n')[-1].lower()
        if 'correct' in last_line or 'correct' in text.lower():
            phase = b"Correct"
        else:
            phase = b"Insert"
            
        return phase, text.encode('utf-8')
    except Exception as e:
        # Fallback to simple heuristic if LLM fails
        phase = b"Insert" if z_vel <= -0.0001 else b"Correct"
        return phase, f"LLM Error: {str(e)}".encode('utf-8')

def annotate_file(file_path):
    print(f"\n--- Processing {os.path.basename(file_path)} ---")
    with h5py.File(file_path, 'a') as f:
        num_steps = f['step'].shape[0]
        
        ee = f['embodiment']['ee']
        left_marker = f['tactile']['left_tactile']['marker']
        right_marker = f['tactile']['right_tactile']['marker']
        
        phases = []
        reasonings = []
        has_inserted = False
        
        for i in range(num_steps):
            start_idx = max(0, i - 4)
            ee_hist = ee[start_idx:i+1]
            left_marker_hist = left_marker[start_idx:i+1]
            right_marker_hist = right_marker[start_idx:i+1]
            
            if len(ee_hist) > 1:
                z_vel = ee_hist[-1][2] - ee_hist[0][2]
                left_disp = np.mean(np.linalg.norm(left_marker_hist[-1] - left_marker_hist[0], axis=-1))
                right_disp = np.mean(np.linalg.norm(right_marker_hist[-1] - right_marker_hist[0], axis=-1))
                mean_marker_disp = max(left_disp, right_disp)
            else:
                z_vel = 0.0
                mean_marker_disp = 0.0
                
            # Initial phase uses robust Python heuristics
            if not has_inserted:
                is_downward = z_vel <= -0.0001
                # Ignore the first 25 frames because the robot adjusting its grip on the plug causes a massive tactile spike
                is_contact = mean_marker_disp >= 0.15 and i >= 25
                
                if is_downward and is_contact:
                    phase = b"Insert"
                    has_inserted = True
                elif is_downward:
                    phase = b"Align"
                else:
                    phase = b"Move"
                    
                reasoning = f"Heuristics: Z-vel={z_vel:.4f}, Marker={mean_marker_disp:.4f}. is_downward={is_downward}, is_contact={is_contact}".encode('utf-8')
            else:
                # To dramatically speed up execution and reduce LLM flickering, only call the LLM every 5 frames
                if i % 5 == 0 or i == num_steps - 1 or 'last_llm_phase' not in locals():
                    phase, reasoning = evaluate_correct_vs_insert(ee_hist, left_marker_hist, right_marker_hist)
                    last_llm_phase = phase
                    last_llm_reasoning = reasoning
                else:
                    phase = last_llm_phase
                    reasoning = last_llm_reasoning
                    
            phases.append(phase)
            reasonings.append(reasoning)
            
            # Print a debug log when state changes to easily verify
            if i == 0 or (len(phases) > 1 and phases[-1] != phases[-2]) or i == num_steps - 1:
                print(f"Frame {i:03d}: {phase.decode('utf-8')} | Z-vel: {z_vel:.4f}, Marker Disp: {mean_marker_disp:.4f}")
                
        if 'annotation' not in f:
            f.create_group('annotation')
        if 'phase' in f['annotation']:
            del f['annotation']['phase']
        if 'reasoning' in f['annotation']:
            del f['annotation']['reasoning']
            
        f['annotation'].create_dataset('phase', data=np.array(phases))
        # Ensure we properly encode the list of reasonings into hdf5
        dt = h5py.string_dtype(encoding='utf-8')
        f['annotation'].create_dataset('reasoning', data=np.array([r.decode('utf-8') for r in reasonings], dtype=object), dtype=dt)
        print(f"Successfully saved annotations to {file_path}")

def main(dataset_dir):
    hdf5_files = glob.glob(os.path.join(dataset_dir, '*.hdf5'))
    print(f"Found {len(hdf5_files)} HDF5 files in {dataset_dir}")
    hdf5_files.sort()
    for file_path in hdf5_files:
        annotate_file(file_path)

if __name__ == '__main__':
    target_dir = '/home/khuzema/UniVTAC/eval_result/ACT/insert_HDMI/deploy_policy_insert_HDMI/2026-04-16_15:03:40/hdf5'
    main(target_dir)
