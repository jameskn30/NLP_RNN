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


# Cheatsheet

## Reinforcement Learning Algorithm Families

### 1. Value-Based Methods
**Examples:** DQN, Double DQN, Dueling DQN, Rainbow
- **Core idea:** Learn the value of states/actions
- **Pros:**
  - Sample efficient for discrete action spaces
  - Clear objective function (minimizing TD error)
  - Can be more stable with proper implementation
- **Cons:**
  - Cannot naturally handle continuous action spaces
  - May require significant replay buffer memory
  - Often requires careful exploration strategy

### 2. Policy Gradient Methods
**Examples:** REINFORCE, VPG, TRPO
- **Core idea:** Directly optimize policy parameters
- **Pros:**
  - Can learn stochastic policies
  - Naturally handles continuous action spaces
  - More straightforward objective (maximize expected return)
- **Cons:**
  - Often high variance in gradient estimates
  - Sample inefficient
  - Can converge to local optima

### 3. Actor-Critic Methods
**Examples:** A2C/A3C, PPO, SAC, DDPG, TD3
- **Core idea:** Combine value function learning with policy optimization
- **Pros:**
  - Reduced variance compared to pure policy gradient
  - Can handle continuous action spaces
  - Often more stable than pure policy gradients
- **Cons:**
  - More complex implementation
  - Two sets of parameters to optimize
  - Balancing actor and critic learning can be tricky

### 4. Model-Based RL
**Examples:** MBPO, PETS, MuZero
- **Core idea:** Learn a model of the environment for planning/simulation
- **Pros:**
  - Sample efficient (can plan without real interactions)
  - Can generalize better with good models
  - Works well when model is accurate
- **Cons:**
  - Model errors can compound ("model bias")
  - Complex implementation
  - Computationally expensive for planning

## Key Terminology

- **MDP (Markov Decision Process)**: Mathematical framework for RL with states, actions, transitions, rewards
- **Value Functions**:
  - **V(s)**: Expected return starting from state s
  - **Q(s,a)**: Expected return taking action a in state s
- **Policy (π)**: Strategy for selecting actions in states
- **Return**: Sum of discounted rewards
- **Discount Factor (γ)**: Weight for future rewards (0-1)
- **Exploration vs Exploitation**: Balance between trying new actions and using known good actions
- **On-policy vs Off-policy**:
  - **On-policy**: Learn about the policy being followed
  - **Off-policy**: Learn about a different policy than the one being followed
- **Experience Replay**: Store and reuse past experiences for training
- **Temporal Difference (TD) Learning**: Learn from incomplete episodes using bootstrapping
- **Advantage Function (A)**: How much better an action is compared to average (Q(s,a) - V(s))
- **Entropy Regularization**: Encouraging exploration by maximizing policy entropy

## Special Techniques

- **Prioritized Experience Replay**: Sample important transitions more frequently
- **Hindsight Experience Replay (HER)**: Learn from failures by pretending they were successes
- **Curiosity-Driven Exploration**: Generate intrinsic rewards for novel states
- **Curriculum Learning**: Train on progressively harder tasks
- **Imitation Learning**: Learn from expert demonstrations
- **Multi-Agent RL**: Multiple agents learning simultaneously in shared environment


