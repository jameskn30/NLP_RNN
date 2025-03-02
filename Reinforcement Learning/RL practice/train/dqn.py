import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
import torch
import os
import multiprocessing

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set number of parallel environments
num_cpu = multiprocessing.cpu_count()
print('CPU num count = ', num_cpu)
num_envs = max(1, min(4, num_cpu // 2))  # Use at most 4 or half the cores
print(f"Using {num_envs} environments with {num_cpu} CPU cores available")

# Create vectorized environment
env_id='LunarLander-v3',
env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
env = VecMonitor(env)  # Add monitoring wrapper
