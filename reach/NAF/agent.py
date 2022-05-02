import os
import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from NAF.model import Policy
from replay_buffer import replay_buffer
import her

def MSELoss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class NAF_AGENT(object):
    def __init__(self, env, log_dir, writer=None):

        self.base_logdir = log_dir
        # get the environment params
        self.observation_dim = env.observation_space['observation'].shape[0]  # 10
        self.goal_dim = env.observation_space['achieved_goal'].shape[0]  # 3
        self.action_dim = env.action_space.shape[0]  # 4
        self.action_bound = env.action_space.high[0]
        self.num_inputs = env.observation_space['observation'].shape[0] + env.observation_space["achieved_goal"].shape[0]# + env.observation_space["desired_goal"].shape[0]

        # ==============agent params================
        self.global_step = 0 # record times of train() was called
        self.GAMMA = 0.99
        self.lr = 0.001
        self.tau = 0.95  # the smaller tau is, the faster target net be updated
        self.batch_size = 256
        self.buffer_size = 10000000
        self.her = True
        self.replay_k = 4 # her params
        self.replay_strategy = 'future' # her mode
        self.clip_range = 5
        # ==========================================
        # replay buffer
        self.her_module = her.her_sampler(self.replay_strategy, self.replay_k, env.compute_reward)
        if self.her == True:
            self.buffer = replay_buffer(
                {'obs': self.observation_dim, 'goal': self.goal_dim, 'action': self.action_dim, 'max_timesteps': 50},
                self.buffer_size, self.her_module.sample_her_transitions)
        else:
            self.buffer = replay_buffer(
                {'obs': self.observation_dim, 'goal': self.goal_dim, 'action': self.action_dim, 'max_timesteps': 50},
                self.buffer_size, None)

        # tensorboard writer
        self.writer = writer  # tensorboard writer

        self.model = Policy(self.num_inputs, self.action_dim)
        self.target = Policy(self.num_inputs, self.action_dim)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        hard_update(self.target, self.model)



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
        torch.save(self.model.state_dict(), os.path.join(weights_dir, prex + 'model.pkl'))
        torch.save(self.target.state_dict(), os.path.join(weights_dir, prex + 'target.pkl'))


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
            self.model.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'model.pkl')))
            self.target.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'target.pkl')))

        except Exception as e:
            print(e)
            print('load weights filed!')
            exit()
        print("Load weights successful")

    def get_action(self, state, greedy=False, param_noise=None):
        self.model.eval()
        mu, _, _ = self.model((Variable(state), None))
        self.model.train()
        mu = mu.data
        if not greedy:
            mu += torch.Tensor(np.random.standard_normal(mu.shape))

        return mu.clamp(-self.action_bound, self.action_bound)

    def update(self):
        # Update the target networks:
        with torch.no_grad():
            soft_update(self.target, self.model, self.tau)


    def train(self):
        self.global_step += 1
        data = self.buffer.sample(self.batch_size)

        obs, acs, rewards, ags, goals, next_obs = data['obs'], data['actions'], data['r'], data['ag'], data['g'], data['obs_next']
        # in the sample, we already set the target goal and the
        g_next = goals

        # Convert to tensor
        obs = torch.as_tensor(obs, dtype=torch.float)
        acs = torch.as_tensor(acs, dtype=torch.float)
        goals = torch.as_tensor(goals, dtype=torch.float)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float)
        rewards = torch.as_tensor(rewards, dtype=torch.float)
        next_goals = torch.as_tensor(g_next, dtype=torch.float)

        # concentrate the obs and goals
        state = torch.cat([obs, goals], dim=1)
        next_state = torch.cat([next_obs, next_goals], dim=1)

        _, _, next_state_values = self.target((next_state, None))
        # Set y_i = r_i + gamma*V'(x_t+1 | Q')
        expected_state_action_values = rewards + (self.GAMMA * next_state_values)

        # Update Q by minimizing the loss
        # =============Loss=====================
        _, state_action_values, _ = self.model((state, acs))
        loss = MSELoss(state_action_values, expected_state_action_values)

        # =============update weights==================
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the param to target nets
        if self.global_step % 50 == 0:
            self.update()

        # tensorboard
        if self.global_step % 100 == 99 and self.writer != None:
            self.writer.add_scalar('loss', loss, global_step=self.global_step)


