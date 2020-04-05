import numpy as np


class ReplayBuffer:
    def __init__(self, replay_length=0, enable_padding=False):
        self.replay_length = replay_length
        self.enable_padding = enable_padding

        # buffers
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def store(self, state, action, next_state, reward):
        self.tmp_episode_buff.append(
            (state, action, next_state, reward))

    def done_episode(self):
        states, actions, next_states, rewards = zip(*self.tmp_episode_buff)
        states = list(states)
        actions = list(actions)
        next_states = list(next_states)
        rewards = list(rewards)
        episode_len = len(states)
        if self.enable_padding:
            usable_episode_len = episode_len
        else:
            usable_episode_len = episode_len - (self.replay_length - 1)
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
        episode_length = len(states)
        p = slice(idx_in_episode, min(idx_in_episode + self.replay_length, episode_length))
        states, actions, next_states, rewards = states[p], actions[p], next_states[p], rewards[p]
        mask = [True] * len(states)
        if self.enable_padding:
            for i in range(len(states), self.replay_length):
                states.append(states[-1])
                actions.append(actions[-1])
                next_states.append(next_states[-1])
                rewards.append(rewards[-1])
                mask.append(False)
        return states, actions, next_states, rewards, mask

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def mean_reward(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.concatenate([np.array(reward) for reward in rewards]).mean()

    def mean_return(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.sum(reward) for reward in rewards])
