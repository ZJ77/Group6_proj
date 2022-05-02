import itertools
import os
import random
from collections import deque
from copy import deepcopy
from DDPG.model import *
from replay_buffer import replay_buffer
import her

"""
The agent of DDPG
"""
class DDPG_AGENT(object):

    def __init__(self, env, log_dir, writer=None):

        self.base_logdir = log_dir
        # get the environment params
        self.observation_dim = env.observation_space['observation'].shape[0]  # 10
        self.goal_dim = env.observation_space['achieved_goal'].shape[0]  # 3
        self.action_dim = env.action_space.shape[0]  # 4
        self.action_bound = env.action_space.high[0]
        self.action_l2 = 1.

        # ==============agent params================
        self.clip_obs = 200 # observations clip parm
        self.exploration = 0.2
        self.global_step = 0 # record times of train() was called
        self.GAMMA = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.tau = 0.95  # the smaller tau is, the faster target net be updated
        self.batch_size = 256

        self.buffer_size = 10000000
        self.replay_k = 4 # her params
        self.her = False
        self.replay_strategy = 'future' # her mode
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

        # Create actor network
        self.actor_net = Actor(self.observation_dim + self.goal_dim, self.action_dim, self.action_bound)
        # Create critic network / Q net work
        self.critic_net = Critic(self.observation_dim + self.goal_dim + self.action_dim, 1)

        print(self.actor_net)
        print(self.critic_net)

        # Create target networks
        self.actor_target_net = deepcopy(self.actor_net)
        self.critic_target_net = deepcopy(self.critic_net)

        # Freeze the target nets' auto-grad function,
        # this will accelerate the train process
        for p in self.actor_target_net.parameters():
            p.requires_grad = False
        for p in self.critic_target_net.parameters():
            p.requires_grad = False

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

        # graph to tensorboard
        # writer.add_graph(self.actor_net, torch.randn(self.batch_size, self.observation_dim + self.goal_dim))
        # writer.close()

        # writer.add_graph(self.critic_net, [torch.randn(self.batch_size, self.observation_dim + self.goal_dim),\
        #                                     torch.randn(self.batch_size, self.action_dim)])
        # writer.close()


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
        torch.save(self.actor_net.state_dict(), os.path.join(weights_dir, prex + 'actor_net.pkl'))
        torch.save(self.actor_target_net.state_dict(), os.path.join(weights_dir, prex + 'actor_target_net.pkl'))
        torch.save(self.critic_net.state_dict(), os.path.join(weights_dir, prex + 'critic_net.pkl'))
        torch.save(self.critic_target_net.state_dict(), os.path.join(weights_dir, prex + 'critic_target_net.pkl'))

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
            self.actor_net.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'actor_net.pkl')))
            self.actor_target_net.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'actor_target_net.pkl')))
            self.critic_net.load_state_dict(torch.load(os.path.join(weights_dir, prex + 'critic_net.pkl')))
            self.critic_target_net.load_state_dict(
                torch.load(os.path.join(weights_dir, prex + 'critic_target_net.pkl')))
        except Exception as e:
            print(e)
            print('load weights filed!')
            exit()
        print("Load weights successful")

    def get_action(self, observation, goal, greedy=False):
        """
        This act just use agent to choose action
        :param observation:
        :param goal the goal
        :param greedy: if greedy is True, the agent will not do any exploration
        :return: action
        """
        observation = torch.as_tensor([observation], dtype=torch.float32)
        goal = torch.as_tensor([goal], dtype=torch.float32)
        obs_goals = torch.cat([observation, goal], dim=-1)

        # close auto-grad to accelerate the process
        with torch.no_grad():
            ac = self.actor_net(obs_goals).numpy()
        ac = np.squeeze(ac)
        if greedy:
            return ac
        else:
            ac = np.random.normal(ac, self.exploration)
            ac = np.clip(ac, -self.action_bound, self.action_bound)
            return ac

    def update(self):
        # Update the target networks:
        with torch.no_grad():
            # .mul with _ means replace the orign data rather than create a new tensor
            for parms, parms_target in zip(self.actor_net.parameters(), self.actor_target_net.parameters()):
                parms_target.data.mul_(self.tau)
                parms_target.data.add_((1 - self.tau) * parms.data)

            for parms, parms_target in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
                parms_target.data.mul_(self.tau)
                parms_target.data.add_((1 - self.tau) * parms.data)

    def train(self):
        """
        Train the network one step
        :return: None
        """
        self.global_step += 1
        data = self.buffer.sample(self.batch_size)
        obs, acs, rewards, goals, next_obs = data['obs'], data['actions'], data['r'], data['g'], data['obs_next']
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
        obs_goals = torch.cat([obs, goals], dim=1)
        next_obs_next_goals = torch.cat([next_obs, next_goals], dim=1)

        # ==============Critic Loss===============
        with torch.no_grad():
            next_ac = self.actor_target_net(next_obs_next_goals)
            next_q = self.critic_target_net(next_obs_next_goals, next_ac)
            next_q = next_q.detach()  # TODO confirm this
            # target
            y_target = rewards + self.GAMMA * next_q
            clip_return = 1 / (1 - self.GAMMA)
            y_target = torch.clamp(y_target, -clip_return, 0)

        # predict, come from the critic net. MUST witch GRAD, or can't backward
        q_predict = self.critic_net(obs_goals, acs)
        # loss func
        critic_loss = ((y_target - q_predict) ** 2).mean()

        # =============Acotr Loss=====================
        ac_predict = self.actor_net(obs_goals)
        q = self.critic_net(obs_goals, ac_predict)
        actor_loss = -q.mean()  # attention the "-"
        actor_loss += self.action_l2 * (ac_predict / self.action_bound).pow(2).mean()

        # =============update weights==================
        # Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # sync_grads(self.critic_net)
        self.critic_optimizer.step()

        # Update the param to target nets
        if self.global_step % 50 == 0:
            self.update()

        # tensorboard
        if self.global_step % 100 == 99 and self.writer != None:
            self.writer.add_scalar('actor_loss', actor_loss, global_step=self.global_step)
            self.writer.add_scalar('critic_loss', critic_loss, global_step=self.global_step)
