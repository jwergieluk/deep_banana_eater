import numpy
import random
from unityagents import UnityEnvironment
import torch
import torch.nn
import torch.optim
from collections import deque, namedtuple
from tqdm import tqdm


class Agent2:
    def __init__(self, env, alpha: float, gamma: float, epsilon: float, update_method: str):
        self.dim_s = env.observation_space.n
        self.dim_a = env.action_space.n
        self.Q = numpy.random.randn(self.dim_s, self.dim_a) + 15.0
        self.alpha = alpha
        self.gamma = gamma
        self.episodes_no = 1.0
        self.epsilon0 = epsilon
        if not hasattr(self, update_method):
            raise ValueError(f'Update method {update_method} not implemented')
        self.update_method = getattr(self, update_method)
        self.test_phase = False

    def step(self, state, action, reward, next_state, done):
        if done:
            self.episodes_no += 1.0
        if not self.test_phase:
            self.update_method(state, action, reward, next_state)

    def sarsa(self, state, action, reward, next_state):
        """ Sarsa on-policy Q-table update rule """
        next_action = self.select_action(next_state)
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def sarsamax(self, state, action, reward, next_state):
        """ Sarsamax aka Q-learning off-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def expected_sarsa(self, state, action, reward, next_state):
        """ Expected Sarsa on-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        policy_vector_for_next_state = numpy.repeat(self.epsilon()/self.dim_a, self.dim_a)
        policy_vector_for_next_state[next_action] += 1.0 - self.epsilon()
        self.Q[state, action] = self.Q[state, action] * (1.0 - self.alpha) + self.alpha * (
                reward + self.gamma * numpy.dot(policy_vector_for_next_state, self.Q[next_state, :]))

    def epsilon(self):
        if self.test_phase:
            return 0.0
        n = self.episodes_no
        if n < 11000.0:
            return 0.1
        return self.epsilon0

    def select_action(self, state):
        if random.random() <= self.epsilon():
            return random.randint(0, self.dim_a-1)
        return numpy.argmax(self.Q[state, :])

    def get_policy(self):
        return numpy.argmax(self.Q, axis=1)


class QNet(torch.nn.Module):
    def __init__(self, input_dim: int, action_no):
        super().__init__()
        self._net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, action_no)
        )

    def forward(self, x):
        return self._net(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_space_dim, no_actions, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            no_actions (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.no_actions = no_actions
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(numpy.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(numpy.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(numpy.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(numpy.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(numpy.vstack([e.done for e in experiences if e is not None]).astype(numpy.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent0:
    LEARNING_RATE = 0.001
    UPDATE_EVERY = 120
    REPLAY_BUFFER_SIZE = 1024
    BATCH_SIZE = 32
    GAMMA = 0.99

    def __init__(self, state_space_dim: int, no_actions: int, device):
        self.no_actions = no_actions
        self.state_space_dim = state_space_dim

        self.q_net = QNet(self.state_space_dim, self.no_actions)
        self.q_net.to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.LEARNING_RATE)
        self.loss = torch.nn.MSELoss()

        self._replay_buffer = ReplayBuffer(self.state_space_dim, self.no_actions, self.REPLAY_BUFFER_SIZE, self.BATCH_SIZE)
        self.t = 1

    def epsilon(self):
        return 0.1

    def get_action(self, state):
        if random.random() <= self.epsilon():
            return random.randint(0, self.no_actions-1)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(state)
        self.q_net.train()
        return numpy.argmax(action_values.cpu().detach().numpy())

    def learn(self, state: numpy.ndarray, action, reward, next_state, done):
        self.t += 1
        self._replay_buffer.add(state, action, reward, next_state, done)

        if self.t % self.UPDATE_EVERY != 0:
            return
        if len(self._replay_buffer) < self.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self._replay_buffer.sample()

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.q_net(states).gather(1, actions)

        # Compute loss
        loss_value = self.loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     


class UnityEnvWrapper:
    """ This class provides gym-like wrapper around the unity environment """

    def __init__(self, env_file: str = 'Banana_Linux_NoVis/Banana.x86_64'):
        self._env = UnityEnvironment(file_name=env_file)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        env_info = self._env.reset(train_mode=True)[self._brain_name]
        state = env_info.vector_observations[0]

        self.state_space_dim = len(state)
        self.action_space_size = self._brain.vector_action_space_size

    def reset(self, train_mode: bool = False):
        env_info = self._env.reset(train_mode)[self._brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self._env.step(action)[self._brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        return next_state, reward, done, None

    def close(self):
        self._env.close()


def train(env, agent, max_episodes: int = 2):
    for episode in range(max_episodes):
        state = env.reset(train_mode=True)
        score = 0
        for i in tqdm(range(1000), ascii=True):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if done:
                break


def test(env: UnityEnvWrapper, agent):
    state = env.reset(train_mode=False)
    score = 0
    for i in range(5):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        print('Action: ', action, 'Reward: ', reward)

        score += reward
        state = next_state
        if done:
            break


def main(device):
    env = UnityEnvWrapper()
    agent = Agent0(env.state_space_dim, env.action_space_size, device)
    train(env, agent)

    #test(env, agent)
    env.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)

