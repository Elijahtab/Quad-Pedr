# RL-Based Navigation for Unitree Go2

This repository implements a hierarchical reinforcement learning framework for the Unitree Go2 quadruped. It provides custom environments for rough terrain locomotion and high-level navigation, designed to run as an extension within NVIDIA Isaac Lab.

## Prerequisites
This project requires a fully functioning installation of **NVIDIA Isaac Lab.** We will omit a step-by-step guide to installation as NVIDIA themselves provides a simple installation guide.
- **System Requirements:** Firstly, please verify your hardware system meets the [Isaac Sim Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- **Installation:** Follow the official [Issac Lab Instruction Guide w/ Pip](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

**Verification:** To confirm the installation is successful please run the examples. 


## Installation
Once Isaac Lab is installed, you can now install this project as an extension. 

### Step 1: Clone the Repo
Navigate to the `source/extensions` directory inside of your Isaac Lab installation folder. *(If the extensions folder doesn't exist, create it)*

```bash
# Example path; adjust to your actual install location
cd ~/path/to/issaclab/source/
mkdir -p extensions
cd extensions

# Clone this repository
git clone https://github.com/Elijahtab/CS-175-Project.git go2_navigation
```

### Step 2: Install Dependencies
Use Python executable provided by Isaac Lab to install the package.
```bash
cd go2_navigation

# Install this package
../../isaaclab.sh -p -m pip install -e .
```
*Note: The `-e` flag allows you to edit the code and see changes immediately without reinstalling.*

## Usage 
All commands should be **run from the root of your Isaac Lab Installation.**

### DEMO: Play Pre-Trained Policy
To verify the installation and see the robot in action, run the command below. This serves as the **project demo**, loading our pre-trained model and generating a video of the agent navigating.
```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task=Isaac-Nav-Flat-Go2-Play-v0 \
  --num_envs 1 \
  --video --video_length 1000 \
  --checkpoint source/extensions/go2_navigation/trained_models/flat_policy.pt
```
*Note: The `--checkpoint` flag points directly to the .pt file included in this repo. To continue to use this to get videos, please replace the `.pt` path.*

**Where is the video?** The resulting video will be saved automatically to: `logs/rsl_rl/Isaac-Nav-Flat-Go2-Play-v0/videos/`

### Training from Scratch
To train the policies yourself it'll be like this:
#### Train Flat Baseline:
```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task=Isaac-Nav-Go2-Flat-v0 --headless
```
#### Train Rough Terrain:
```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task=Isaac-Nav-Go2-v0 --headless
```
#### Train Hierarchical Navigation (Flat):
```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task=Isaac-Nav-Flat-Go2-v0 --headless
```
***Note on Model Saving:** This specfied training script saves logs and checkpoints to `/isaaclab/logs/rsl_rl/{task_name}/{date}` If you wish to update the pre-trained models in this repo you'll need to copy and paste the `.pt` file to the trained_model directory*

## Repository Structure
- `navigation_env_cfg.py:` High-level navigation environment configuration.
- `rough_env_cfg.py:` Low-level locomotion configuration for rough terrain.
- `custom_rewards.py:` Custom reward functions for gait and stability.
- `custom_obs.py:` Custom observations, including LiDAR and goal vectors.
- `custom_events.py:` Logic for spawning static obstacles and randomizing the domain.
- `agents/:` Configuration files for the PPO agents (RSL-RL).

## Common Errors
- **Import Errors:** Ensure you ran the `pip install -e .` command using `./isaaclab.sh -p` wrapper, not just the system Python
- **Simulator Crashes:**  Ensure you have the proper drivers as required by the latest Isaac Sim.