from collections import deque

import numpy as np
import torch
from tqdm.auto import trange, tqdm
from unityagents import UnityEnvironment

from agent import Agent


class Trainer:
    def __init__(self, env_path, args, num_workers=1, num_agents=20, termination_threshold=30, n_trajectories=1e6,
                 max_t=1e4):
        self.num_workers = num_workers
        self.termination_threshold = termination_threshold
        self.max_t = max_t
        self.n_trajectories = n_trajectories
        self.num_agents = num_agents
        self.env_path = env_path
        self.args = args

        self.scores = None

        self.env, self.state_size, self.action_size, self.brain_name = self.create_env()
        self.processes = []

    # TODO: Harvest experiences with multiprocessing (Possible attempt for A3C)
    def train(self):
        self.masterprocess()

    def create_env(self):
        env = UnityEnvironment(file_name=self.env_path)
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        env_info = env.reset()[brain_name]

        action_size = brain.vector_action_space_size

        states = env_info.vector_observations
        state_size = states.shape[1]

        return env, state_size, action_size, brain_name

    def masterprocess(self):
        env, state_size, action_size = self.env, self.state_size, self.action_size
        agent = Agent(state_size, action_size, number_of_agents=self.num_agents, is_master=True, args=self.args, device="cpu")

        scores_deque = deque(maxlen=100)
        scores = []
        tqdm_bar = trange(1, self.n_trajectories, desc="Trajectories")
        episode_bar = tqdm(total=self.max_t)
        train_mode = True
        for i in tqdm_bar:

            state = env.reset(train_mode=train_mode)[self.brain_name].vector_observations
            score = 0
            for t in range(self.max_t):
                action, prob, q_value = agent.act(state[0])
                action2, prob2, q_value2 = agent.act(state[1])
                env_info = env.step([action.detach().cpu().data.numpy(),
                                     action2.detach().cpu().data.numpy()])[self.brain_name]
                next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
                agent.step(action, reward, prob, done, q_value)
                state = next_state
                score += np.mean(reward)
                episode_bar.set_description("Time Step T: {}, Score: {:.2f}".format(t, score))
                episode_bar.update()
                # if done:
                #     break

            episode_bar.reset()
            tqdm_bar.set_description("Episode: {}, Score: {:.2f}".format(i, score))
            scores_deque.append(score)
            scores.append(score)
            # train_mode = score < 10.0
            if i % 100 == 0:
                torch.save(agent.TwoHeadModel.state_dict(), 'checkpoint.pth')

                if np.mean(scores_deque) > self.termination_threshold:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100,
                                                                                                 np.mean(                                                                           scores_deque)))
                    break

        self.scores = scores
        env.close()


if __name__ == '__main__':
    args = {
        "lr": 1e-4,
        "buffer_size": 5,
        "batch_size": 5,
        "gamma": .99,
        "tau": 1e-3,
    }
    trainer = Trainer(env_path="Tennis_Linux_NoVis/Tennis.x86_64",
                      args=args,
                      num_workers=5,
                      num_agents=2,
                      termination_threshold=30,
                      n_trajectories=int(2000),
                      max_t=1000)

    trainer.train()