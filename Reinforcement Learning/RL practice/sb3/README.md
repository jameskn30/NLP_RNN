### stable baselines study

Read about RL and Stable Baselines3

Do quantitative experiments and hyperparameter tuning if needed

Evaluate the performance using a separate test environment (remember to check wrappers!)

For better performance, increase the training budget

---

## RL Algorithm Selection Guide

| Algorithm | State Space | Action Space | When to Use | Considerations |
|-----------|------------|--------------|-------------|----------------|
| DQN       | Discrete/Continuous | Discrete only | - Simple environments with discrete actions<br>- When you need sample efficiency | - Can't handle continuous actions<br>- Needs careful tuning of exploration parameters |
| PPO       | Discrete/Continuous | Discrete/Continuous | - General-purpose algorithm<br>- When stability matters more than sample efficiency<br>- Good first algorithm to try | - Generally robust<br>- Works well across many environments<br>- Slower than SAC for some continuous control tasks |
| SAC       | Continuous | Continuous | - Continuous control problems<br>- When sample efficiency matters<br>- When exploration is important | - State-of-the-art for continuous control<br>- Adds entropy for better exploration<br>- Not suitable for discrete action spaces |
| A2C/A3C   | Discrete/Continuous | Discrete/Continuous | - When parallelization is possible<br>- When you need a simpler alternative to PPO | - Less sample efficient than PPO<br>- Benefits from distributed training |
| DDPG      | Continuous | Continuous | - Continuous control problems<br>- When you need deterministic policies | - Often less stable than SAC or TD3<br>- No exploration by default (needs noise) |
| TD3       | Continuous | Continuous | - Continuous control problems<br>- When addressing overestimation bias is important | - Improved version of DDPG<br>- More stable but more complex |
| HER       | Discrete/Continuous | Discrete/Continuous | - Sparse reward environments<br>- Goal-based tasks | - Used as extension to other algorithms (DQN, SAC, etc.)<br>- Dramatically improves sample efficiency in sparse rewards |

### Quick Selection Guide:

- **Discrete actions only**: DQN
- **Continuous actions**: SAC (most sample-efficient), PPO (more stable)
- **General purpose**: PPO (works well in most cases)
- **Sparse rewards**: Try algorithms with HER
- **Need for exploration**: SAC has built-in exploration (entropy)

