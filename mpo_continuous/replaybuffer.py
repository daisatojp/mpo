import numpy as np


class ReplayBuffer:
    def __init__(self, pick_trajectory=False, backward_length=0):
        self.pick_trajectory = pick_trajectory
        self.backward_length = backward_length

        # buffer
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def store(self, state, action, next_state, reward):
        self.tmp_episode_buff.append(
            (state, action, next_state, reward))

    def done_episode(self):
        states, actions, next_states, rewards = zip(*self.tmp_episode_buff)
        episode_len = len(states)
        usable_episode_len = episode_len - self.backward_length
        self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
        self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
        self.episodes.append((states, actions, next_states, rewards))
        self.tmp_episode_buff = []

    def clear(self):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def __getitem__(self, idx):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        idx_in_episode = idx - start_idx
        states, actions, next_states, rewards = self.episodes[episode_idx]
        if self.pick_trajectory:
            p = slice(idx_in_episode, idx_in_episode+self.backward_length)
        else:
            p = idx_in_episode
        state, action, next_state, reward = states[p], actions[p], next_states[p], rewards[p]
        return state, action, next_state, reward

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def mean_reward(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean(rewards)
