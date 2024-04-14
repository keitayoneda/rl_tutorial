from dataclasses import dataclass
import torch
from torch import nn
from typing import List
import gym
from collections import deque
import numpy as np
import argparse


@dataclass
class Experience:
    state: object
    action: object
    reward: float
    next_state: object
    done: bool

class ValueFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.model(state)

class Agent:
    def __init__(self, load_path=None):
        self.value_function:ValueFunction = ValueFunction()
        if load_path:
            self.load(load_path)

    def load(self, load_path):
        self.value_function.load_state_dict(torch.load(load_path))


    def save(self, save_path):
        torch.save(self.value_function.state_dict(), save_path)
    
    def policy(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        pred = self.value_function.forward(state)
        action = torch.argmax(pred).item()
        return action
    

    def update(self, exps, gamma:float,  optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module):
        states_tensor = torch.tensor([exp.state for exp in exps])
        pred = self.value_function.forward(states_tensor)
        targets = pred.clone().detach()
        for i, exp in enumerate(exps):
            v_real = exp.reward
            if not exp.done:
                v_real += gamma * torch.max(self.value_function.forward(exp.next_state)).item()
            targets[i][exp.action] = v_real

        loss = loss_fn(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class Trainer:
    def __init__(self, env: gym.Env, agent: Agent, gamma: float, lr: float=1e-3):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(agent.value_function.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def train(self, episodes: int, max_steps: int):
        exps = deque(maxlen=128)
        batch_size = 32
        for e in range(episodes):
            state = self.env.reset()[0]
            print(f"===Episode {e} start===")
            reward_sum = 0
            step_num = 0
            for step_num in range(max_steps):
                if step_num % 100 == 0:
                    print(f"Step {step_num}")
                # if e % 10 == 0:
                #     self.env.render()
                action = self.agent.policy(state, epsilon=0.3)
                next_state, reward, info, done, _ = self.env.step(action)
                reward = 0


                # round angle
                while state[2] > np.pi:
                    state[2] -= 2*np.pi
                while state[0] < -np.pi:
                    state[2] += 2*np.pi

                # 位置
                position_threshold = 2.0
                if abs(state[0]) < position_threshold:
                    reward += 0.0
                else:
                    reward += -(abs(state[0]) - position_threshold)*3.0
                if abs(state[0]) >= 3.0:
                    reward += -200
                # 速度
                if abs(state[1]) > 1.0:
                    reward +=  - abs(state[1])
                # 角度
                angle_threshold = 0.5
                if abs(state[2]) > angle_threshold:
                    reward += -abs(state[2])*3
                else:
                    reward += (angle_threshold - abs(state[2]))*5
                # 角速度
                reward += -(abs(state[3]))*2
                # 終了
                if step_num == max_steps-1 and abs(state[0]) < 0.3:
                    reward += 100
                if step_num == max_steps-1 and abs(state[0]) > 0.3:
                    reward += -100

                
                reward_sum += reward
                exp = Experience(state, action, reward, next_state, done)
                exps.append(exp)
                if len(exps) >= batch_size:
                    sample_exps = np.random.choice(exps, batch_size, replace=False)
                    self.agent.update(sample_exps, self.gamma, self.optimizer, self.loss_fn)
                if abs(state[0]) > 3.0:
                    print(f"state: {state}")
                    break
                state = next_state
            print(f"Episode {e} ended. Reward: {reward_sum/step_num}")
            if e % 10 == 0:
                self.agent.save(f"model2/value_base_{e}.pth")


def train(model_path=None):
    env = gym.make("CartPole-v1")
    if model_path:
        agent = Agent(model_path)
    else:
        agent = Agent()
    trainer = Trainer(env, agent, gamma=0.90, lr=1e-4)
    trainer.train(1000, 1000)
    agent.save("model2/value_base.pth")

def play(model_path):
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent(model_path)
    state = env.reset()[0]
    for _ in range(1000):
        action = agent.policy(state, epsilon=0.0)
        state, reward, info, done, _ = env.step(action)
        if done or abs(state[0]) > 2.4:
            break

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "train":
        if args.model_path:
            train(model_path=args.model_path)
        else:
            train()
    else:
        play(args.model_path)

