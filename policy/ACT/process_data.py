import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import h5py
import numpy as np
import argparse
import json
from tqdm import tqdm
from envs.utils.data import HDF5Handler


ANNOTATED_CONFIG_NAMES = {"annotated", "expert_demos", "skill_annotated"}
PHASE_TO_SKILL_ID = {
    "Move": 0,
    "Align": 1,
    "Insert": 2,
    "Correct": 3,
}


def _decode_phase_label(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_phase_label(value.item())
    return str(value)


def _phase_labels_to_ids(phase_labels):
    phase_ids = []
    unknown = set()
    for label in phase_labels:
        decoded = _decode_phase_label(label)
        if decoded not in PHASE_TO_SKILL_ID:
            unknown.add(decoded)
            continue
        phase_ids.append(PHASE_TO_SKILL_ID[decoded])
    if unknown:
        raise ValueError(
            f"Unknown phase labels encountered: {sorted(unknown)}. "
            f"Known labels: {sorted(PHASE_TO_SKILL_ID)}"
        )
    return np.asarray(phase_ids, dtype=np.int64)


def resolve_input_path(task_name, task_config, explicit_input_path=None):
    if explicit_input_path is not None:
        input_path = Path(explicit_input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        return input_path

    default_path = Path("../../data") / task_name / task_config
    if default_path.exists():
        return default_path

    if task_config in ANNOTATED_CONFIG_NAMES:
        annotated_path = Path("../../data_annotated/expert_demos")
        if annotated_path.exists():
            return annotated_path

    raise FileNotFoundError(
        f"Could not resolve dataset path for task='{task_name}', config='{task_config}'. "
        f"Tried: {default_path}"
    )


def detect_phase_labels(path):
    hdf5_dir = Path(path) / "hdf5"
    if not hdf5_dir.exists():
        hdf5_dir = Path(path)
    first_file = next(iter(sorted(hdf5_dir.glob("*.hdf5"))), None)
    if first_file is None:
        return False
    with h5py.File(first_file, "r") as f:
        return "annotation" in f and "phase" in f["annotation"]


def load_hdf5(dataset_paths, camera_type, downsample_factor, include_skill_labels=False):
    data_paths = [
        'embodiment/joint',
    ]
    if camera_type == 'all':
        data_paths.append(f'observation/head/rgb')
        data_paths.append(f'observation/wrist/rgb')
    else:
        data_paths.append(f'observation/{camera_type}/rgb')    
    
    with h5py.File(str(dataset_paths[0]), 'r') as f:
        try:
            f['tactile/left_tactile/rgb_marker']
            data_paths.append('tactile/left_tactile/rgb_marker')
            data_paths.append('tactile/right_tactile/rgb_marker')
        except:
            data_paths.append('tactile/left_gsmini/rgb_marker')
            data_paths.append('tactile/right_gsmini/rgb_marker')
        if include_skill_labels and 'annotation' in f and 'phase' in f['annotation']:
            data_paths.append('annotation/phase')

    data = HDF5Handler().batch_gather_hdf5(
        dataset_paths,
        data_paths=data_paths,
        resize=False,
        convert_channels=False,
        downsample_factor=downsample_factor,
    )
 
    return data, data_paths


def data_transform(path, episode_num, save_path, include_skill_labels=False):
    hdf5_dir = Path(path) / 'hdf5'
    if not hdf5_dir.exists():
        hdf5_dir = Path(path)
        if len(list(hdf5_dir.glob('*.hdf5'))) == 0:
            print(f"HDF5 directory does not exist at \n{hdf5_dir}\n")
            raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    
    # 获取所有 episode 文件
    hdf5_files = sorted(hdf5_dir.glob('*.hdf5'), key=lambda x: int(x.stem))
    assert episode_num <= len(hdf5_files), f"data num not enough: requested {episode_num}, found {len(hdf5_files)}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    global task_name
    with open('../task_settings.json', 'r') as f:
        task_settings = json.load(f)
    assert task_name in task_settings, f"Task '{task_name}' not found in task_settings.json"
    camera_type = task_settings[task_name].get('camera_type', 'head')
    downsample_factor = task_settings[task_name].get('downsample', 1)
    print(f"Loading {episode_num} episodes with camera type '{camera_type}', downsample factor {downsample_factor}.")

    # 批量加载所有 episode
    dataset_paths = [str(hdf5_files[i]) for i in range(episode_num)]
    data, data_paths = load_hdf5(
        dataset_paths[:episode_num],
        camera_type,
        downsample_factor,
        include_skill_labels=include_skill_labels,
    )
    
    # 提取批量数据
    joint_state_all = data['embodiment/joint_state'][:, 0:8]  # (T_total, 8)
    joint_action_all = data['embodiment/joint_action'][:, 0:8]  # (T_total, 8)
    if camera_type == 'all':
        head_cam_all = data[f'observation/head/rgb']  # (T_total, H, W, 3)
        wrist_cam_all = data[f'observation/wrist/rgb']  # (T_total, H, W, 3)
    else:
        head_cam_all = data[f'observation/{camera_type}/rgb']  # (T_total, H, W, 3)
    
    # Use the detected tactile paths from load_hdf5
    left_tac_key = next(p for p in data_paths if 'left' in p and 'rgb_marker' in p)
    right_tac_key = next(p for p in data_paths if 'right' in p and 'rgb_marker' in p)
    left_tac_all = data[left_tac_key]
    right_tac_all = data[right_tac_key]
    episode_ends = data['episode_ends']
    skill_all = data.get('annotation/phase')
    
    start_idx = 0
    for i in tqdm(range(episode_num), desc='Writing episodes'):
        end_idx = episode_ends[i]
        
        joint_state = joint_state_all[start_idx:end_idx]
        joint_action = joint_action_all[start_idx:end_idx]
        if camera_type == 'all':
            head_cam = head_cam_all[start_idx:end_idx]
            wrist_cam = wrist_cam_all[start_idx:end_idx]
        else:
            head_cam = head_cam_all[start_idx:end_idx]
        left_tac = left_tac_all[start_idx:end_idx]
        right_tac = right_tac_all[start_idx:end_idx]
        skill_labels = None
        if skill_all is not None:
            skill_labels = _phase_labels_to_ids(skill_all[start_idx:end_idx])

        # 保存为 ACT 格式的 HDF5
        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(joint_action))
            if skill_labels is not None:
                f.create_dataset("skill", data=skill_labels, dtype=np.int64)
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(joint_state))
            image = obs.create_group("images")
            # 只保存头部相机和触觉传感器
            if camera_type == 'all':
                image.create_dataset("cam_high", data=np.stack(head_cam), dtype=np.uint8)
                image.create_dataset("cam_wrist", data=np.stack(wrist_cam), dtype=np.uint8)
            else:
                image.create_dataset("cam_high", data=np.stack(head_cam), dtype=np.uint8)
            image.create_dataset("tac_left", data=np.stack(left_tac), dtype=np.uint8)
            image.create_dataset("tac_right", data=np.stack(right_tac), dtype=np.uint8)
        start_idx = end_idx

    if include_skill_labels:
        label_map_path = Path(save_path) / "skill_label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(PHASE_TO_SKILL_ID, f, indent=4)

    return episode_num, camera_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TacArena episodes for ACT training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument("task_config", type=str, help="Task config (e.g., demo)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional explicit path to raw HDF5 episodes.",
    )

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    input_path = resolve_input_path(task_name, task_config, explicit_input_path=args.input_path)
    include_skill_labels = task_config in ANNOTATED_CONFIG_NAMES or detect_phase_labels(input_path)
    output_path = f"./data/sim-{task_name}/{task_config}-{expert_data_num}"
    
    begin, cam_type = data_transform(
        input_path,
        expert_data_num,
        output_path,
        include_skill_labels=include_skill_labels,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "tac_left", "tac_right"] if cam_type != 'all' \
            else ["cam_high", "cam_wrist", "tac_left", "tac_right"],
    }
    if include_skill_labels:
        SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"]["skill_label_map"] = PHASE_TO_SKILL_ID

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
    
