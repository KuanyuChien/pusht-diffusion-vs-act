"""Check that the lerobot env works on a GPU node: GPU, offline dataset, simulator, official policy."""

import time

import torch

# 1. GPU
print("== 1. GPU")
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
t = time.time()
for _ in range(50):
    y = x @ x
torch.cuda.synchronize()
print(f"50 matmuls (4096x4096) in {time.time() - t:.3f}s")

# 2. Dataset, read offline (HF_HUB_OFFLINE=1), decode one video frame with torchcodec + ffmpeg
print("== 2. Dataset")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Explicit root: offline, LeRobot only finds a dataset at the root you give it (or HF_LEROBOT_HOME/<repo_id>)
ds = LeRobotDataset("lerobot/pusht", root="/scratch/general/vast/u1470293/pusht/data/lerobot/pusht")
frame = ds[100]
print("episodes:", ds.num_episodes, "| frames:", ds.num_frames)
print("image:", tuple(frame["observation.image"].shape), "| state:", frame["observation.state"].tolist(),
      "| action:", frame["action"].tolist())

# 3. Simulator, headless (SDL_VIDEODRIVER=dummy)
print("== 3. Simulator")
import gymnasium as gym
import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, info = env.reset(seed=0)
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("pixels:", obs["pixels"].shape, "| agent_pos:", obs["agent_pos"].tolist(), "| reward:", round(float(reward), 3))
env.close()

# 4. Official policy, loaded from local cache onto the GPU
print("== 4. Official policy")
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to("cuda")
n_params = sum(p.numel() for p in policy.parameters())
print(f"loaded lerobot/diffusion_pusht: {n_params / 1e6:.1f}M parameters on {next(policy.parameters()).device}")

print("== ALL CHECKS PASSED")
