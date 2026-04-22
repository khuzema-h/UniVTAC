<h1 align="center">UniVTAC</h1>

> UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking<br>
> [arXiv](https://arxiv.org/abs/2602.10093) | [PDF](https://arxiv.org/pdf/2602.10093) | [Website](https://univtac.github.io/) | [HuggingFace Dataset](https://huggingface.co/datasets/byml/UniVTAC) | [Modelscope Dataset](https://modelscope.cn/datasets/byml2024/UniVTAC)

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on top of **NVIDIA Isaac Lab** and **TacEx (UIPC-based tactile simulation)**. It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across a diverse suite of contact-rich manipulation tasks — all with high-fidelity tactile feedback from simulated GelSight Mini, ViTai GF225, or XenseWS sensors.

## Installation

See the [Installation Guide](./docs/Installation.md) for detailed setup instructions, including installing the environment, installing TacEx from the modified local source and setting up cuRobo for motion planning.

## Task Gallery

UniVTAC currently includes the following manipulation tasks, all featuring tactile sensing:

| Task | Module | Description |
|---|---|---|
| **Collect** | `collect` | Collect contact-rich tactile data for pretraining |
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle off a surface near a wall |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |

## Data Collection

See the [Data Collection Guide](./docs/Collection.md) for instructions on how to run the automated data collection pipeline, configure task-specific parameters, and understand the output data structure.

Dataset containing 100 episodes per task can be downloaded from [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC), [Modelscope](https://modelscope.cn/datasets/byml2024/UniVTAC) or by running the script in `data/download.sh`.

## Train & Eval Policies

UniVTAC includes several baseline policies implemented under the `policy/` directory:

- ACT: Action Chunking with Transformers with/without tactile inputs
- Abation: ACT ablation variants for modality comparison
- ViTAL: ACT with CLIP-pretrained tactile-vision encoders in ViTAL

Each policy is a self-contained module under `policy/` with its own data processing, training, and deployment scripts. All policies share a unified evaluation entry point at the project root:

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
```

### Quick Start: Training & Evaluating ACT
If you want to train and evaluate the ACT policy on a task like `insert_HDMI`:

1.  **Download & Link data**: 
    ```bash
    # Download specific task dataset
    bash data/download.sh "insert_HDMI/**"
    # Link the downloaded HDF5 files to the training directory
    ln -s ~/.cache/modelscope/hub/datasets/byml2024/UniVTAC/master/insert_HDMI/demo/*.hdf5 data/insert_HDMI/demo/
    ```

2.  **Process Data**:
    ```bash
    cd policy/ACT
    # Usage: bash process_data.sh <task> <config> <num_episodes>
    bash process_data.sh insert_HDMI demo 100
    ```

3.  **Train**:
    ```bash
    # Usage: bash train.sh <task> <config> <num_episodes> <seed> <gpu_id>
    bash train.sh insert_HDMI demo 100 0 0
    ```

4.  **Evaluate**:
    ```bash
    cd ../..
    # Use the deployment config we created
    bash eval_policy.sh insert_HDMI demo ACT/deploy_policy_insert_HDMI 0
    ```

### Quick Start: Training & Evaluating SkillACT
If you want to train and evaluate the skill-augmented ACT policy on `insert_HDMI` using annotated phase labels:

1. **Prepare annotated data**:
   - Place the annotated raw HDF5 files under `data_annotated/expert_demos/`.
   - The current preprocessing expects per-timestep skill annotations at `annotation/phase`.

2. **Process annotated data**:
   ```bash
   cd policy/ACT
   python process_data.py insert_HDMI annotated 100
   ```

3. **Train SkillACT**:
   ```bash
   # Usage: bash train.sh <task> <config> <num_episodes> <seed> <gpu_id> <train_config>
   bash train.sh insert_HDMI annotated 100 0 0 skill_train_config
   ```

4. **Evaluate SkillACT**:
   ```bash
   cd ../..
   bash eval_policy.sh insert_HDMI demo ACT/deploy_policy_insert_HDMI 0 --policy_variant SkillACT
   ```

The default `insert_HDMI` deploy config is already set up to use:
- `policy/ACT/act_ckpt/act-insert_HDMI/annotated-100/skill_train_config`
- `policy_best.ckpt`

If you want to evaluate regular ACT instead, omit `--policy_variant SkillACT`.

### Vision-only (head + wrist cameras, no tactile)
If you want to train/evaluate a policy using **only RGB cameras** (head + wrist) and **no tactile**:

1. **Prepare processed ACT data that includes both cameras**:
   - Your processed ACT dataset must contain `cam_high` **and** `cam_wrist`.
   - `process_data.py` uses `policy/task_settings.json` to decide whether to export only head (`camera_type: head`) or both head+wrist (`camera_type: all`) for a task.

2. **Train ACT vision-only** using the provided config:

```bash
cd policy/ACT
# train_config_vision_all.yml uses camera_names: [cam_high, cam_wrist] and tactile_names: []
bash train.sh insert_HDMI demo 100 0 0 train_config_vision_all
```

3. **Evaluate vision-only** (no tactile observations, force both cameras):

```bash
cd ../..
bash eval_policy.sh insert_HDMI demo ACT/deploy_policy_insert_HDMI 0 --vision_only --camera_type all
```

### Passing additional arguments to scripts
- **ACT training** (`policy/ACT/train.sh`) forwards any extra args after the 6th positional argument to `imitate_episodes.py`.
  - Example:

```bash
cd policy/ACT
bash train.sh insert_HDMI demo 100 0 0 train_config_vision_all --wandb_mode offline
```

- **Evaluation** (`eval_policy.sh`) forwards extra args after the GPU id to `scripts/eval_policy.py`.
  - Example:

```bash
bash eval_policy.sh insert_HDMI demo ACT/deploy_policy_insert_HDMI 0 --total_num 20 --record
```


For parallel evaluation over many seeds:

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

The evaluation results, including videos and success rate logs, will be saved in the `eval_result/` directory under the project root.

To deploy your own policy, refer to the [Deploy Your Policy](./docs/Deploy.md).

## TODO

- Data collection and evaluation are now only supported on the GelSight Mini sensor. We will add support for ViTai GF225 and XenseWS in the near future.

## 👍 Citations
If you find our work useful, please consider citing:

```
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

## Contact
<div style="text-align: center;">
  <img src="https://box.nju.edu.cn/seafhttp/f/fc1021a908ff49309f22/?op=view" alt="Wechat Group" width="300"/>
</div>
