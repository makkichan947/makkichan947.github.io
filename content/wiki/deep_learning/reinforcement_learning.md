+++
date = '2025-10-25T10:00:00+08:00'
draft = false
title = '强化学习'
comments = true
weight = 7
+++

# 强化学习

强化学习（Reinforcement Learning, RL）是机器学习的一个重要分支，通过智能体与环境的交互来学习最优策略，在游戏AI、机器人控制、自动驾驶等领域取得了显著成就。

## 🎯 强化学习基础

### 马尔可夫决策过程 (MDP)

强化学习的核心框架是马尔可夫决策过程：

**MDP包含的元素**：
- **状态空间 S**：环境可能的所有状态
- **动作空间 A**：智能体可以执行的所有动作
- **奖励函数 R(s,a)**：从状态s执行动作a后获得的奖励
- **状态转移概率 P(s'|s,a)**：从状态s执行动作a后转移到状态s'的概率
- **折扣因子 γ**：未来奖励的衰减因子

**MDP的目标**：
找到一个策略π(a|s)，使得从初始状态开始的累积折扣奖励最大化：
$$\pi^* = \arg\max_\pi \mathbb{E}[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)]$$

### 强化学习组成

**智能体 (Agent)**：
- 观察环境状态
- 选择和执行动作
- 接收奖励信号
- 学习最优策略

**环境 (Environment)**：
- 接收智能体的动作
- 转移到新状态
- 提供奖励信号
- 提供状态观察

## 🏗️ 强化学习算法

### Q-Learning算法

**Q-Learning**是最经典的基于值函数的强化学习算法：

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q表：状态-动作值函数
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        """ε-贪心策略选择动作"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用

    def learn(self, state, action, reward, next_state, done):
        """Q-Learning更新"""
        # 当前Q值
        current_q = self.q_table[state, action]

        # 目标Q值
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # Q值更新
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 使用Q-Learning
env = gym.make('FrozenLake-v1')
state_size = env.observation_space.n
action_size = env.action_space.n

agent = QLearningAgent(state_size, action_size)

# 训练
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")
```

### 深度Q网络 (DQN)

**DQN**使用深度神经网络来近似Q函数：

```python
import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """构建DQN网络"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def update_target_model(self):
        """更新目标网络"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """ε-贪心策略"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 使用DQN
import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

# 训练
episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            agent.update_target_model()
            print(f"Episode: {e}, Score: {total_reward}")
            break
```

### 策略梯度方法

**REINFORCE算法**：

```python
class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        """构建策略网络"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model

    def choose_action(self, state):
        """根据策略选择动作"""
        state = np.reshape(state, [1, self.state_size])
        probs = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs

    def learn(self, states, actions, rewards):
        """REINFORCE学习"""
        discounted_rewards = self._discount_rewards(rewards)

        with tf.GradientTape() as tape:
            # 计算策略概率
            states = np.array(states)
            probs = self.model(states)

            # 选择对应动作的概率
            action_probs = tf.gather(probs, actions, axis=1, batch_dims=1)

            # 计算损失（负对数似然）
            loss = -tf.reduce_mean(tf.math.log(action_probs) * discounted_rewards)

        # 计算梯度并更新
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss.numpy()

    def _discount_rewards(self, rewards):
        """计算折扣奖励"""
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0

        for i in reversed(range(len(rewards))):
            running_sum = running_sum * self.gamma + rewards[i]
            discounted_rewards[i] = running_sum

        # 标准化
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 1e-8

        return discounted_rewards

# 使用REINFORCE
agent = REINFORCEAgent(state_size, action_size)

# 训练
episodes = 1000
for episode in range(episodes):
    states = []
    actions = []
    rewards = []

    state = env.reset()
    done = False

    while not done:
        action, prob = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    loss = agent.learn(states, actions, rewards)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Loss: {loss:.4f}")
```

## 🎮 高级强化学习算法

### Actor-Critic方法

```python
class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.99

        # Actor网络（策略）
        self.actor = self._build_actor()
        # Critic网络（价值）
        self.critic = self._build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_actor(self):
        """构建Actor网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(24, activation='relu')(inputs)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_critic(self):
        """构建Critic网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(24, activation='relu')(inputs)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def choose_action(self, state):
        """选择动作"""
        state = np.reshape(state, [1, self.state_size])
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs

    def learn(self, state, action, reward, next_state, done):
        """Actor-Critic学习"""
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        # Critic更新
        with tf.GradientTape() as tape:
            value = self.critic(state)
            next_value = self.critic(next_state) if not done else 0

            target = reward + self.gamma * next_value
            critic_loss = tf.reduce_mean(tf.square(target - value))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Actor更新
        with tf.GradientTape() as tape:
            probs = self.actor(state)
            action_prob = probs[0][action]

            advantage = target - value
            actor_loss = -tf.math.log(action_prob) * advantage

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return critic_loss.numpy(), actor_loss.numpy()
```

### 近端策略优化 (PPO)

```python
class PPOAgent:
    def __init__(self, state_size, action_size, clip_ratio=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.clip_ratio = clip_ratio
        self.gamma = 0.99
        self.learning_rate = 0.0003

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_actor(self):
        """构建Actor网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_critic(self):
        """构建Critic网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_action(self, state):
        """获取动作和概率"""
        state = np.reshape(state, [1, self.state_size])
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]

    def compute_advantages(self, rewards, values, next_values, dones):
        """计算优势函数"""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * 0.95 * next_non_terminal * last_gae_lam

        return advantages

    def train(self, states, actions, old_probs, advantages, returns):
        """PPO训练"""
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            values = self.critic(states)

            # 计算策略损失
            new_probs = tf.gather(probs, actions, axis=1, batch_dims=1)
            ratio = new_probs / old_probs
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # 计算价值损失
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # 总损失
            loss = policy_loss + 0.5 * value_loss

        # 更新Actor
        actor_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 更新Critic
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return loss.numpy()
```

## 🎯 应用领域

### 游戏AI

**AlphaGo的启发**：
- **蒙特卡洛树搜索 (MCTS)**：结合深度学习和搜索
- **自我对弈**：通过自我对弈提升策略
- **策略网络和价值网络**：分别预测走法和局面价值

```python
# 简化版AlphaGo策略
class SimpleAlphaGo:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()

    def _build_policy_network(self):
        """策略网络"""
        inputs = tf.keras.Input(shape=(self.board_size, self.board_size, 1))
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.board_size * self.board_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_value_network(self):
        """价值网络"""
        inputs = tf.keras.Input(shape=(self.board_size, self.board_size, 1))
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 机器人控制

**连续控制任务**：
- **DDPG (Deep Deterministic Policy Gradient)**：处理连续动作空间
- **TD3 (Twin Delayed DDPG)**：改进的DDPG算法
- **SAC (Soft Actor-Critic)**：最大化熵的强化学习

```python
class DDPGAgent:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high

        # Actor网络（确定性策略）
        self.actor = self._build_actor()
        # Critic网络（Q函数）
        self.critic = self._build_critic()
        # 目标网络
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()

        # 复制权重到目标网络
        self.update_target_networks(1.0)

    def _build_actor(self):
        """构建Actor网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(400, activation='relu')(inputs)
        x = tf.keras.layers.Dense(300, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='tanh')(x)

        # 缩放到动作范围
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_critic(self):
        """构建Critic网络"""
        state_inputs = tf.keras.Input(shape=(self.state_size,))
        action_inputs = tf.keras.Input(shape=(self.action_size,))

        # 状态路径
        state_out = tf.keras.layers.Dense(400, activation='relu')(state_inputs)
        state_out = tf.keras.layers.Dense(300, activation='relu')(state_out)

        # 动作路径
        action_out = tf.keras.layers.Dense(300, activation='relu')(action_inputs)

        # 合并
        merged = tf.keras.layers.Add()([state_out, action_out])
        outputs = tf.keras.layers.Dense(1)(merged)

        return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=outputs)
```

### 自动驾驶

**自动驾驶中的强化学习**：
- **路径规划**：学习最优驾驶路径
- **行为决策**：在复杂交通环境中决策
- **控制优化**：优化车辆控制参数

```python
class AutonomousDrivingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 感知模块
        self.perception_model = self._build_perception_model()
        # 决策模块
        self.decision_model = self._build_decision_model()
        # 控制模块
        self.control_model = self._build_control_model()

    def _build_perception_model(self):
        """感知模块：处理传感器数据"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        # 检测其他车辆、行人、交通标志等
        outputs = {
            'vehicles': tf.keras.layers.Dense(10, activation='sigmoid')(x),
            'pedestrians': tf.keras.layers.Dense(5, activation='sigmoid')(x),
            'traffic_lights': tf.keras.layers.Dense(3, activation='softmax')(x)
        }
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_decision_model(self):
        """决策模块：做出驾驶决策"""
        inputs = tf.keras.Input(shape=(128,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
```

## 📊 评估和调试

### 评估指标

```python
def evaluate_agent(agent, env, episodes=100):
    """评估智能体性能"""
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean([r > threshold for r in total_rewards])
    }

# 使用评估函数
results = evaluate_agent(agent, env, episodes=100)
print(f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"成功率: {results['success_rate']:.2%}")
```

### 调试技巧

```python
class DebuggingAgent:
    def __init__(self, agent):
        self.agent = agent
        self.episode_rewards = []
        self.q_values = []
        self.gradients = []

    def debug_episode(self, env):
        """调试单个episode"""
        state = env.reset()
        episode_reward = 0
        episode_q_values = []

        while True:
            # 记录Q值
            if hasattr(self.agent, 'model'):
                q_values = self.agent.model.predict(np.reshape(state, [1, -1]), verbose=0)[0]
                episode_q_values.append(np.max(q_values))

            action = self.agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward

            if done:
                break

            state = next_state

        self.episode_rewards.append(episode_reward)
        self.q_values.append(episode_q_values)

        return episode_reward

    def plot_debugging_info(self):
        """绘制调试信息"""
        plt.figure(figsize=(12, 4))

        # 奖励曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')

        # Q值曲线
        plt.subplot(1, 3, 2)
        for i, q_values in enumerate(self.q_values[-10:]):  # 最近10个episode
            plt.plot(q_values, alpha=0.3, label=f'Episode {len(self.episode_rewards)-10+i}')
        plt.xlabel('Step')
        plt.ylabel('Max Q-value')
        plt.title('Q-values')
        plt.legend()

        # 奖励分布
        plt.subplot(1, 3, 3)
        plt.hist(self.episode_rewards, bins=20)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')

        plt.tight_layout()
        plt.show()

# 使用调试工具
debug_agent = DebuggingAgent(agent)

for episode in range(100):
    reward = debug_agent.debug_episode(env)

debug_agent.plot_debugging_info()
```

## 📚 学习资源

### 经典论文
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - DQN
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - DQN Nature
- [Mastering the game of Go with deep neural networks](https://www.nature.com/articles/nature16961) - AlphaGo
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO

### 在线资源
- [OpenAI Gym](https://gym.openai.com/) - 强化学习环境
- [Stable Baselines](https://stable-baselines.readthedocs.io/) - 强化学习算法库
- [RLlib](https://ray.readthedocs.io/en/latest/rllib.html) - Ray中的强化学习库

### 吴恩达课程
- 深度学习课程中关于强化学习的部分

## 🎯 实际项目

### 智能体训练框架

```python
class RLTrainingFramework:
    def __init__(self, env_name, agent_class, config):
        self.env_name = env_name
        self.agent_class = agent_class
        self.config = config

        self.env = gym.make(env_name)
        self.agent = agent_class(self.env.observation_space.shape[0],
                                self.env.action_space.n, **config)

    def train(self, num_episodes, eval_interval=100):
        """训练智能体"""
        best_reward = -np.inf
        rewards_history = []

        for episode in range(num_episodes):
            # 训练episode
            episode_reward = self._train_episode()

            rewards_history.append(episode_reward)

            # 定期评估
            if episode % eval_interval == 0:
                eval_reward = self._evaluate_agent()
                print(f"Episode {episode}: Train Reward = {episode_reward:.2f}, "
                      f"Eval Reward = {eval_reward:.2f}")

                # 保存最佳模型
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save_model(f"best_model_{episode}")

        return rewards_history

    def _train_episode(self):
        """训练单个episode"""
        state = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        return total_reward

    def _evaluate_agent(self, episodes=10):
        """评估智能体"""
        total_reward = 0

        for _ in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / episodes

    def _save_model(self, filename):
        """保存模型"""
        # 实现模型保存逻辑
        pass

# 使用训练框架
config = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995
}

framework = RLTrainingFramework('CartPole-v1', DQNAgent, config)
rewards = framework.train(num_episodes=1000)
```

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*