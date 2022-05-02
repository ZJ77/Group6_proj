import os
import argparse
import shutil
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from NAF.agent import *
from gym.envs.robotics.fetch.reach import FetchReachEnv


ENV = 'FetchReach-v1'
REWARD_TYPE = 'sparse'
EPOCH_NUM = 50
MAX_STEP_PER_EPISODE = 50
TEST_EPISODES = 50
TRAIN_NUMS_AFTER_EP = 50
EPISODES_PER_EPOCH = 10
MEMORY_SIZE = 2000000
UPDATES_PER_STEP = 5
BATCH_SIZE = 256
EXPLORATION = 0.2
BASE_LOG_DIR = os.path.join(os.getcwd(), 'Logs')

class Logger(object):
    def __init__(self, dir, fileName):
        self.path = os.path.join(dir, fileName)

    def log(self, content):
        with open(self.path, mode='a+', encoding='utf-8') as f:
            f.write(content+'\t \n')


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
        obs = observation['observation']
        d_g = observation['desired_goal']
        for _ in range(MAX_STEP_PER_EPISODE):
            with torch.no_grad():
                state = torch.Tensor([np.concatenate((obs, d_g))])
                actions = agent.get_action(state)
                actions = actions.cpu().numpy()
            observation_new, _, done, info = env.step(actions[0])
            obs = observation_new['observation']
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
    #env = gym.make(ENV)
    env = FetchReachEnv(reward_type=REWARD_TYPE)
    if video_record :
        v_dir = os.path.join(logdir, "videos")
        if not os.path.exists(v_dir):
            os.mkdir(v_dir)
        env = gym.wrappers.Monitor(env, directory=v_dir, video_callable= lambda episode_id: episode_id%100==0 or episode_id>490, force = True)
    # create an agent
    agent = NAF_AGENT(env, log_dir=logdir, writer=writer)

    # ====start train the agent ============
    for epoch in range(EPOCH_NUM):
        print("EPOCH {} ||".format(epoch), end='', flush=True)
        for i in range(EPISODES_PER_EPOCH): #
            epoch_obs, epoch_a_gs, epoch_d_gs, epoch_acs, epoch_r = [], [], [], [], []

            print('-{}'.format(i), end='', flush=True)
            episode_obs, episode_a_gs, episode_d_gs, episode_acs, episode_r= [], [], [], [], []
            # ==== start a episode
            ob_all = env.reset()
            ob = ob_all['observation']
            a_g = ob_all['achieved_goal']
            d_g = ob_all['desired_goal']
            for step in range(MAX_STEP_PER_EPISODE):
                # get a action
                # ac = agent.act(ob, d_g)
                with torch.no_grad():
                    state = torch.Tensor([np.concatenate((ob_all["observation"], ob_all["desired_goal"]))])
                    ac = agent.get_action(state, np.random.sample() > EXPLORATION)
                    ac = ac[0].cpu().numpy()

                next_ob_all, r, done, _ = env.step(ac)
                next_ob = next_ob_all['observation']
                next_a_g = next_ob_all['achieved_goal']

                episode_obs.append(ob)
                episode_a_gs.append(a_g)
                episode_d_gs.append(d_g)
                episode_acs.append(ac)
                episode_r.append([r])

                ob = next_ob
                a_g = next_a_g

            # process the last ob and a_g
            episode_obs.append(ob)  # list 50
            episode_a_gs.append(a_g)

            epoch_obs.append(episode_obs)
            epoch_acs.append(episode_acs)
            epoch_a_gs.append(episode_a_gs)
            epoch_d_gs.append(episode_d_gs)
            epoch_r.append(episode_r)

            epoch_obs = np.asarray(epoch_obs)
            epoch_acs = np.asarray(epoch_acs)
            epoch_a_gs = np.asarray(epoch_a_gs)
            epoch_d_gs = np.asarray(epoch_d_gs)
            epoch_r = np.asarray(epoch_r)
            # store
            agent.buffer.store_episode([epoch_obs, epoch_a_gs, epoch_d_gs, epoch_acs, epoch_r])

            # Train
            for _ in range(TRAIN_NUMS_AFTER_EP):
                agent.train()

        acc = evaluate(env,agent)
        print('|| +acc: {}'.format(acc))
        writer.add_scalar('test success rate/epoch', acc, epoch)
        # save weights
        if acc >0.9 :
            agent.save_weights(epoch)
            logger.log("epoch id: {}, acc: {}".format(epoch, acc))


def test(episodes = 10, agent = None):
    """
    This func is used to test the agent performence and give a 3D view
    :return: None
    """
    #env = gym.make(ENV)
    env = FetchReachEnv(reward_type=REWARD_TYPE)
    if agent ==None:
        logdir = input("(Test Mode)Please input the logdir name: ")
        exp_id = input("(Test Mode)Please input the weights id: ")

        # Without the logdir, Print error and exit
        logdir = os.path.join(BASE_LOG_DIR, logdir)

        if not os.path.exists(logdir):
            print("Sorry, this {} doesn't exist".format(logdir))
            exit()

        # create an agent:
        agent = NAF_AGENT(env, log_dir=logdir)
        # load weights
        agent.load_weights(exp_id)
        print(evaluate(env,agent))

    obs_all = env.set()
    ob = obs_all['observation']
    d_g = obs_all['desired_goal']
    for _ in range(episodes):
        for step in range(MAX_STEP_PER_EPISODE):
            env.render()
            state = torch.Tensor([np.concatenate((ob, d_g))])
            ac = agent.get_action(state, True)
            ac = ac[0].cpu().numpy()
            obs_all, r , done, info = env.step(ac)
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
