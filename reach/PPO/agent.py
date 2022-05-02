import os
import torch
from PPO.model import ActorCritic

"""
The agent of PPO
"""

def MSELoss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()

class PPO_AGENT(object):

    def __init__(self, env, log_dir, writer=None):

        self.base_logdir = log_dir
        # get the environment params
        self.observation_dim = env.observation_space['observation'].shape[0]  # 10
        self.goal_dim = env.observation_space['achieved_goal'].shape[0]  # 3
        self.action_dim = env.action_space.shape[0]  # 4
        self.action_bound = env.action_space.high[0]
        self.action_std = 0.5

        # ==============agent params================
        self.GAMMA = 0.99
        self.lr = 0.001
        self.K_epochs = 50
        self.eps_clip = 0.2

        # tensorboard writer
        self.writer = writer  # tensorboard writer

        self.policy = ActorCritic(self.observation_dim + self.goal_dim, self.action_dim, self.action_std)
        self.policy_old = ActorCritic(self.observation_dim + self.goal_dim, self.action_dim, self.action_std)

        print(self.policy)

        # optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)


    def save_weights(self, prex=None):
        """
        Save the net weights
        :param prex: the id of epoch
        :return: None
        """
        if prex != None:
            prex = str(prex) + '_'
        weights_dir = os.path.join(self.base_logdir, 'weights')
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        torch.save(self.policy.state_dict(), os.path.join(weights_dir, prex + 'policy.pkl'))
        torch.save(self.policy_old.state_dict(), os.path.join(weights_dir, prex + 'policy_old.pkl'))


    def load_weights(self, prex=None):
        """
        Load the weights
        :param prex: id of epoch
        :return: None
        """
        if prex != None:
            prex = prex + '_'

        weights_dir = os.path.join(self.base_logdir, 'weights')
        try:
            self.policy.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'policy.pkl')))
            self.policy_old.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'policy_old.pkl')))

        except Exception as e:
            print(e)
            print('load weights filed!')
            exit()
        print("Load weights successful")

    def get_action(self, observation):
        """
        This act just use agent to choose action
        """
        observation = torch.as_tensor([observation], dtype=torch.float32)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(observation)
        return action, action_logprob

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        dis_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.done)):
            if done:
                dis_reward = 0
            dis_reward = reward + (self.GAMMA * dis_reward)
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)# Normalizing the rewards

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.eval(old_states, old_actions)# Evaluating old actions and values

            ratios = torch.exp(logprobs - old_logprobs)# ratio (pi_theta / pi_theta__old)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * MSELoss(state_values, rewards) - 0.01 * dist_entropy
            loss = loss.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())# update policy
