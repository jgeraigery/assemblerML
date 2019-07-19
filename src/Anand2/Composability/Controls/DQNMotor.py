# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from Problems.DCMotor import *

class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = Motor()
	#if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
       # if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=2, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(80, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        x=np.arange(-20,20,0.5)
        return np.random.choice(x,1) if (np.random.random() <= epsilon) else x[np.argmax(self.model.predict(state))]

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 2])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            x=np.arange(-20,20,0.5)
            index=np.where(x==action)
            y_target[0][index] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=1000)

        for e in range(self.n_episodes):
            print("New Run Begins",e)
            self.env.reset()
            state = np.round(self.preprocess_state(self.env.state),decimals=3)
            done = False
            ref_state=np.array(state)
            ref_state[:,0]=2
            i = 0
            for ep in range(100):
                action = self.choose_action(state, self.get_epsilon(e))
                next_state= np.round(self.env.step(action),decimals=3)
                next_state = self.preprocess_state(next_state)
                reward=self.env.getReward(ref_state)
                reward=(np.reshape(reward,[1,1]))
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                print(state,ref_state,action,reward)
            self.replay(self.batch_size)
        

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
