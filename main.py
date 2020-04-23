from collections import deque

from tqdm.auto import tqdm, trange
from unityagents import UnityEnvironment
import numpy as np

from agent import CompetetiveAgent


class MADDPGTrainer:
    def __init__(self, env_path, args, num_workers=1, termination_threshold=30, n_episodes=1e6,
                 max_t=1e3):
        self.num_workers = num_workers
        self.termination_threshold = termination_threshold
        self.max_t = max_t
        self.n_episodes = n_episodes
        self.env_path = env_path
        self.args = args

        self.scores = None

        self.env, self.num_agents, self.action_size, self.state_size, self.brain_name = self.create_env()
        self.processes = []

    def create_env(self):
        env = UnityEnvironment(file_name=self.env_path)
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # number of agents
        num_agents = len(env_info.agents)
        # print('Number of agents:', num_agents)
        # size of each action
        action_size = brain.vector_action_space_size
        # print('Size of each action:', action_size)
        # examine the state space
        states = env_info.vector_observations
        state_size = states.shape[1]
        # print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        # print('The state for the first agent looks like:', states[0])

        return env, num_agents, action_size, state_size, brain_name

    def train(self):
        env, state_size, action_size = self.env, self.state_size, self.action_size

        agent = CompetetiveAgent(state_size, action_size, self.num_agents, args=self.args,
                                 device="cuda:0")

        scores_deque = deque(maxlen=100)
        scores = []
        tqdm_bar = trange(1, self.n_episodes, desc="Episodes")
        episode_bar = tqdm(total=self.max_t)
        train_mode = True
        add_noise = True
        for i in tqdm_bar:
            agent.reset()
            states = env.reset(train_mode=train_mode)[self.brain_name].vector_observations
            episode_scores = np.zeros(self.num_agents)
            loss = []
            for t in range(self.max_t):
                actions = agent.act(states, add_noise=add_noise)
                env_info = env.step(actions)[self.brain_name]
                next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
                curr_loss = agent.step(states, actions, rewards, next_states, dones)
                if curr_loss is not None:
                    loss.append(curr_loss)
                states = next_states
                episode_scores += rewards
                episode_bar.set_description("Time Step T: {}, Score: {:.2f}".format(t, np.max(episode_scores)))
                episode_bar.update()
                if np.any(dones):
                    break

            episode_bar.reset()

            if len(loss) > 0:
                if add_noise and loss[-1][0] > .2:
                    add_noise = False
                tqdm_bar.set_description("Episode: {}, Score: {:.2f}, Critic Loss: {:.2f}, Actor Loss: {:.2f}".format(i, np.max(episode_scores), loss[-1][0], loss[-1][1]))
            scores_deque.append(episode_scores)
            scores.append(scores)
            if i % 100 == 0:

                if np.mean(scores_deque) > self.termination_threshold:
                    agent.save()
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100,
                                                                                                 np.mean(scores_deque)))
                    break

        self.scores = scores
        env.close()


if __name__ == '__main__':
    args = {
        "memory_size": 1e6,
        "batch_size": 512,
        "discount_factor": .99,
        "tau": .2,
        "lr_actor": 1e-4,
        "lr_critic": 1e-4,
        "weight_decay": 0,
    }
    print(args)

    trainer = MADDPGTrainer(env_path="Tennis_Windows_x86_64/Tennis.exe",
                            args=args,
                            termination_threshold=.5,
                            n_episodes=int(20000),
                            max_t=10000)

    trainer.train()
