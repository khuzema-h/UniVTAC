import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT, SkillACT
# from act_policy import ACT
from torchvision import transforms

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        self.policy_variant = args.get("policy_variant", "ACT")
        if self.policy_variant not in {"ACT", "SkillACT"}:
            raise ValueError(f"Unsupported ACT policy variant: {self.policy_variant}")

        self.ep_num = args.get('ep_num', os.environ.get('EP_NUM', '100'))

        train_config_names = args.get("train_config_names", {})
        self.train_config_name = args.get(
            'train_config_name',
            train_config_names.get(self.policy_variant, os.environ.get('TRAIN_CONFIG', 'train_config')),
        )
        ckpt_names = args.get("ckpt_names", {})
        self.ckpt_name = args.get("ckpt_name", ckpt_names.get(self.policy_variant, "policy_last.ckpt"))
        
        # Prioritize ckpt_dir from args (from YAML) if provided
        ckpt_dirs = args.get("ckpt_dirs", {})
        selected_ckpt_dir = args.get('ckpt_dir')
        if selected_ckpt_dir is None and self.policy_variant in ckpt_dirs:
            selected_ckpt_dir = ckpt_dirs[self.policy_variant]

        if selected_ckpt_dir is not None:
            ckpt_dir = Path(selected_ckpt_dir)
        else:
            ckpt_dir = Path(__file__).parent / "act_ckpt" / f"act-{args['task_name']}" / f"{args['task_config']}-{self.ep_num}" / self.train_config_name

 
        self.task_name = args['task_name']
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = args.get("camera_type") or task_settings[self.task_name].get('camera_type', 'head')
        self.use_tactile = bool(args.get("use_tactile", True))
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")
        print(f"Using tactile: {self.use_tactile}")

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            'ckpt_name': self.ckpt_name,
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        model_cls = SkillACT if self.policy_variant == "SkillACT" else ACT
        self.model = model_cls(train_config)
        print(f"{self.policy_variant} policy loaded from {ckpt_dir / self.ckpt_name}")

    def encode_obs(self, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            return img

        if self.camera_type == 'all':
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = camera_transform(observation["observation"][self.camera_type]["rgb"])
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:8]

        ret = {
            "cam_high": cam_high,
            "qpos": qpos.cpu().numpy()
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        if self.use_tactile:
            # Some tasks/sensors may use different keys; keep existing default.
            left_tac = tactile_transform(observation["tactile"]["left_tactile"]["rgb_marker"])
            right_tac = tactile_transform(observation["tactile"]["right_tactile"]["rgb_marker"])
            ret["tac_left"] = left_tac
            ret["tac_right"] = right_tac
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if task.take_action_cnt % 10 == 0:
            self.save(observation, task.take_action_cnt)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, observation, t):
        from PIL import Image
        
        # Create output directory
        save_dir = Path(f"eval_frames/{self.task_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Map observational frames to save
        frames = {
            'head': observation["observation"]["head"]["rgb"],
            'wrist': observation["observation"]["wrist"]["rgb"],
            'tactile_left': observation["tactile"]["left_tactile"]["rgb_marker"],
            'tactile_right': observation["tactile"]["right_tactile"]["rgb_marker"]
        }

        for name, tensor in frames.items():
            img = Image.fromarray(tensor.cpu().numpy().astype('uint8'))
            img.save(save_dir / f"step_{t:04d}_{name}.png")
