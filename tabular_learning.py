import gym
import collections
import numpy as np
import torch
import torch.nn as nn

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent():
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)


    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
    
    def calculate_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        
        for target_state, count in target_counts.items():
            reward = self.rewards[(state, action, target_state)]
            val = reward + GAMMA * self.values[target_state]
            action_value += val * (count / total)
        
        return action_value 

    def select_action(self, state):
        n_actions = self.env.action_space.n
        action_values = list(map(lambda a: self.calculate_action_value(state, a), range(n_actions)))
        best_action_value = max(action_values)
        return action_values.index(best_action_value)
    
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        is_done = False

        while not is_done:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            state = new_state
        
        return total_reward

    def value_iteration(self):
        n = self.env.action_space.n
        for state in range(self.env.observation_space.n):
            state_values = list(map(lambda a: self.calculate_action_value(state, a), range(n)))
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    iter_num = 0
    best_reward = 0.0
    while best_reward <= 0.80:
        iter_num += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        
        reward = 0.0
        
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        
        reward /= TEST_EPISODES
        
        if reward > best_reward:
            print("Best reward updated: {} to {}".format(best_reward, reward))
            best_reward = reward
        
    print("Solved in {} iterations".format(iter_num))



