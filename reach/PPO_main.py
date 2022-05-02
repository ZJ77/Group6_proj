import os
import argparse
import shutil
import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PPO.agent import PPO_AGENT
from PPO.memory import Memory
from gym.envs.robotics.fetch.reach import FetchReachEnv

ENV = 'FetchReach-v1'
REWARD_TYPE = 'sparse'
EPOCH_NUM = 50
MAX_STEP_PER_EPISODE = 50
TEST_EPISODES = 50
EPISODES_PER_EPOCH = 10
UPDATE_T = 500
BASE_LOG_DIR = os.path.join(os.getcwd(), 'Logs')


class Logger(object):
    def __init__(self, dir, fileName):
        self.path = os.path.join(dir, fileName)

    def log(self, content):
        with open(self.path, mode='a+', encoding='utf-8') as f:
            f.write(content + '\t \n')


def evaluate(env, agent):
    """
    To envalue the acc
    :param env: env
    :param agent: agent
    :return:acc
    """
    total_success_rate = []
    for _ in range(TEST_EPISODES):
        per_success_rate = []
        observation = env.reset()
        ob = observation['observation']
        d_g = observation['desired_goal']
        for _ in range(MAX_STEP_PER_EPISODE):
            with torch.no_grad():
                obs_goals = np.concatenate([ob, d_g])
                ac, logprob = agent.get_action(obs_goals)
            observation_new, _, done, info = env.step(ac.data.numpy().flatten())
            ob = observation_new['observation']
            d_g = observation_new['desired_goal']
            per_success_rate = info['is_success']
            if done:
                break
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate)
    return local_success_rate


def train(video_record=False):
    """
    Train the agent
    :return:
    """
    # customize the log dir name
    dirname = input("(Train Mode)Please input the log dir name:")
    logdir = os.path.join(BASE_LOG_DIR, dirname)
    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    # create a logger
    logger = Logger(logdir, 'weights_list.txt')
    # create tensorboard writer
    writer = SummaryWriter(os.path.join(logdir, 'tb_log'))
    # make env
    # env = gym.make(ENV)
    env = FetchReachEnv(reward_type=REWARD_TYPE)
    if video_record:
        v_dir = os.path.join(logdir, "videos")
        if not os.path.exists(v_dir):
            os.mkdir(v_dir)
        env = gym.wrappers.Monitor(env, directory=v_dir,
                                   video_callable=lambda episode_id: episode_id % 100 == 0 or episode_id > 490,
                                   force=True)
    # create an agent
    memory = Memory()
    agent = PPO_AGENT(env, log_dir=logdir, writer=writer)
    t = 0

    # ====start train the agent ============
    for epoch in range(EPOCH_NUM):
        print("EPOCH {} ||".format(epoch), end='', flush=True)
        for i in range(EPISODES_PER_EPOCH):  #
            print('-{}'.format(i), end='', flush=True)
            # ==== start a episode ====
            ob_all = env.reset()
            ob = ob_all['observation']
            d_g = ob_all['desired_goal']

            for step in range(MAX_STEP_PER_EPISODE):
                t += 1
                # get a action
                with torch.no_grad():
                    obs_goals = np.concatenate([ob, d_g])
                    ac, logprob = agent.get_action(obs_goals)

                next_ob_all, reward, done, _ = env.step(ac.data.numpy().flatten())
                next_ob = next_ob_all['observation']
                ob = next_ob
                # store
                memory.store(torch.as_tensor([obs_goals], dtype=torch.float32), ac, logprob, reward, done)

                if t % UPDATE_T == 0:
                    agent.update(memory)
                    memory.clear()
                    t = 0
                if done:
                    break
        acc = evaluate(env, agent)
        print('|| +acc: {}'.format(acc))
        writer.add_scalar('test success rate/epoch', acc, epoch)
        # save weights
        agent.save_weights(epoch)
        logger.log("epoch id: {}, acc: {}".format(epoch, acc))


def test(episodes=10, agent=None):
    """
    This func is used to test the agent performence and give a 3D view
    :return: None
    """
    # env = gym.make(ENV)
    env = FetchReachEnv(reward_type=REWARD_TYPE)
    if agent == None:
        logdir = input("(Test Mode)Please input the logdir name: ")
        exp_id = input("(Test Mode)Please input the weights id: ")

        # Without the logdir, Print error and exit
        logdir = os.path.join(BASE_LOG_DIR, logdir)

        if not os.path.exists(logdir):
            print("Sorry, this {} doesn't exist".format(logdir))
            exit()

        # create an agent:
        agent = PPO_AGENT(env, log_dir=logdir)
        # load weights
        agent.load_weights(exp_id)
        print(evaluate(env, agent))

    obs_all = env.set()
    ob = obs_all['observation']
    d_g = obs_all['desired_goal']
    for _ in range(episodes):
        for step in range(MAX_STEP_PER_EPISODE):
            env.render()
            obs_goals = np.concatenate([ob, d_g])
            ac, logprob = agent.get_action(obs_goals)
            obs_all, r, done, info = env.step(ac.data.numpy().flatten())
            next_ob = obs_all["observation"]
            d_g = obs_all['desired_goal']
            ob = next_ob
            if done:
                print("step:{}".format(step))
                break

        obs_all = env.set_next()
        ob = obs_all['observation']
        d_g = obs_all['desired_goal']


def main():
    # prepare a parser
    parser = argparse.ArgumentParser(description='Train Or Test')
    # add argument to parser.
    parser.add_argument("--test", action="store_true", default=False, help="Test the Agent")
    parser.add_argument("--train", action="store_true", default=False, help="Train the Agent")

    args = parser.parse_args()

    if args.test:
        test(200)

    elif args.train:
        train()
    else:
        f = input("Please input the mode you want to run (train/test, default test): ")
        if f == "train":
            train()
        else:
            test(200)


if __name__ == '__main__':
    main()
