import h5py
import numpy as np
import ollama
import base64
import io
from PIL import Image

VLM_MODEL = 'qwen2.5vl:7b'
RESOLVER_MODEL = 'mistral:7b-instruct'

def depth_to_base64(depth_array):
    min_val = np.min(depth_array)
    max_val = np.max(depth_array)
    if max_val > min_val:
        norm = (depth_array - min_val) / (max_val - min_val) * 255.0
    else:
        norm = np.zeros_like(depth_array)
    img = Image.fromarray(norm.astype(np.uint8)).convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

file_path = '/home/khuzema/UniVTAC/eval_result/ACT/insert_HDMI/deploy_policy_insert_HDMI/2026-04-16_15:03:40/hdf5/1000000.hdf5'

with h5py.File(file_path, 'r') as f:
    left_depth = f['tactile']['left_tactile']['depth'][:]
    right_depth = f['tactile']['right_tactile']['depth'][:]
    ee = f['embodiment']['ee'][:]
    joint = f['embodiment']['joint'][:]

for i in [10, 150]:  # test early and middle frames
    print(f"--- Frame {i} ---")
    left_img_b64 = depth_to_base64(left_depth[i])
    right_img_b64 = depth_to_base64(right_depth[i])
    
    vlm_prompt = (
        "You are analyzing tactile depth sensor readings from a robotic gripper's left and right fingers. "
        "The task is picking up an HDMI plug and inserting it. "
        "Describe the state of the contact based on these two images (left and right finger depth). "
        "Is the gripper holding an object?"
    )
    
    print("Calling VLM...")
    try:
        vlm_response = ollama.chat(
            model=VLM_MODEL,
            messages=[{
                'role': 'user',
                'content': vlm_prompt,
                'images': [left_img_b64, right_img_b64]
            }]
        )
        vlm_text = vlm_response['message']['content']
        print("VLM Output:", vlm_text)
    except Exception as e:
        vlm_text = f"VLM Error: {e}"
        print(vlm_text)
        
    start_idx = max(0, i - 4)
    ee_hist = ee[start_idx:i+1]
    
    resolver_prompt = f"""
You are determining the current phase of a robotic manipulation task.
The task is picking up an HDMI plug, moving it, aligning it with a port, and inserting it.

The 4 possible phases are:
1. Pickup
2. Move
3. Align
4. Insert

Here is the visual language model's analysis of the tactile sensors at the current timestep:
"{vlm_text}"

Recent End-Effector Positions (last {len(ee_hist)} steps):
{np.round(ee_hist, 3).tolist()}

Based on this information, what is the current phase? 
Respond with EXACTLY ONE word from the following list: [Pickup, Move, Align, Insert].
"""
    
    print("Calling Resolver...")
    try:
        res_response = ollama.chat(
            model=RESOLVER_MODEL,
            messages=[{
                'role': 'user',
                'content': resolver_prompt
            }]
        )
        print("Resolver Output:", res_response['message']['content'].strip())
    except Exception as e:
        print("Resolver Error:", e)
