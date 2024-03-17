import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# CartPole 환경 설정
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 신경망 모델 정의
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 강화 학습 알고리즘 정의 (DQN)
class DQN:
    def __init__(self):
        self.model = QNetwork()
        self.target_model = QNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(action_size))
        else:
            with torch.no_grad():
                return self.model(state).argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)  # Change dtype to float32

        Q_targets_next = self.target_model(next_states).max(1)[0].detach()
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.model(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(Q_expected, Q_targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# DQN 알고리즘으로 CartPole 학습
dqn_agent = DQN()
num_episodes = 5000
max_timesteps = 5000
target_score = 500

episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    score = 0
    for t in range(max_timesteps):
        action = dqn_agent.select_action(torch.tensor(state, dtype=torch.float32))
        next_state, reward, done, _ = env.step(action)
        dqn_agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        score += reward

        # 시각화 코드 추가
        env.render()
        if done:
            break
   
    dqn_agent.replay()
    dqn_agent.update_target_model()
    dqn_agent.decay_epsilon()

    episode_rewards.append(score)  # 에피소드별 보상 기록

    print(f"Episode {episode+1}, Score: {score}, Epsilon: {dqn_agent.epsilon}")

    if score >= target_score:
        print(f"Reached target score of {target_score} in episode {episode+1}!")
        break

# 환경 렌더링 종료
env.close()

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards')
plt.show()
