import torch
import numpy as np
import os

from settings import INPUT_MAP
from agent_code.actor_critic import ActorCriticLSTM, ActorCriticMLP
from agent_code.utils import ACTIONS


# Hyperparameter
UPDATE_PPO_AFTER_N_STEPS = 100
MINI_BATCH_SIZE = 25
PPO_EPOCHS_PER_EVALUATION = 8


class PPOAgent:
    def __init__(self, pretrained_model=None, input_feature_size=30, hidden_size=256, network_type='LSTM', device='cuda', train=True):
        self.device = device
        self.model = self._initialize_model(
            pretrained_model, input_feature_size, hidden_size, network_type)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []
        self.loss_sum = 0
        self.n_updates = 0
        self.round_rewards = 0

        self.train = train

    def _initialize_model(self, pretrained_model, input_feature_size, hidden_size, network_type):
        num_outputs = len(INPUT_MAP)
        if pretrained_model:
            model_path = os.path.join('./models', pretrained_model)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Pretrained model at {model_path} not found.")
            return self._load_model(model_path, input_feature_size, hidden_size, num_outputs, network_type)

        return self._create_new_model(input_feature_size, hidden_size, num_outputs, network_type)

    def _load_model(self, model_path, input_feature_size, hidden_size, num_outputs, network_type):
        print("Using pretrained model")
        model = self._select_model(
            network_type, input_feature_size, hidden_size, num_outputs)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device)

    def _create_new_model(self, input_feature_size, hidden_size, num_outputs, network_type):
        print("Using new model")
        return self._select_model(network_type, input_feature_size, hidden_size, num_outputs).to(self.device)

    def _select_model(self, network_type, input_feature_size, hidden_size, num_outputs):
        if network_type == 'LSTM':
            return ActorCriticLSTM(input_feature_size, hidden_size, num_outputs)
        elif network_type == 'MLP':
            return ActorCriticMLP(input_feature_size, hidden_size, num_outputs)
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    @staticmethod
    def compute_gae(next_value, rewards, masks, values, gamma=0.95, tau=0.95):
        """
        Compute General Advantage Estimataion for a sequence of states rewards and value estimates.
        Estimate the advantages of taking actions in a policy

            Parameter:
                next_value: estimated value of the next state
                rewards (list[float]): rewards received at each time step during an episode
                masks (list[bool]): binary masks that indicate whether a state is terminal (0) or not (1)
                values (list): estimated values for each state encountered during the episode
                gamma (float): discount factor
                tau (float): controls the trade-off between bias and variance in the advantage estimates. Higher -> reduces variance

            Return:  1
                returns: list of GAE values for current time step 

        Author: Luke Voss
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        """
        Update step of the Proximal Policy Optimization (PPO) algorithm. 
        This function takes a set of experiences, including states, actions, 
        log probabilities, returns, and advantages, and performs multiple epochs of policy updates

            Parameters:
                ppo_epochs (int): number of PPO update epochs to perform on a mini batch
                mini_batch_size (int):  size of the mini-batches used during the policy update
                states: array containing the states encountered during interactions.
                actions:  array containing the actions taken in response to the states
                log_probs: array containing the logarithm of the probability of taking the actions under the current policy
                returns: array containing the estimated returns (GAE values) for the actions taken
                advantages: array containing the advantages of taking the actions, which guide the policy update
                clip_param: controls the clipping of the policy update. It determines how much the new policy can deviate from the old policy.
                    A smaller clip_param leads to more conservative updates

        Author: Luke Voss
        """
        for i in range(ppo_epochs):
            mean_loss = 0
            batch_size = states.size(0)
            n_updates = ppo_epochs*(batch_size // mini_batch_size)
            for state, action, old_log_probs, return_, advantage in self._generate_batches(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                    1.0 + clip_param) * advantage
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                mean_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if n_updates > 0:
                mean_loss = mean_loss/n_updates

        return mean_loss

    @staticmethod
    def _generate_batches(mini_batch_size, states, actions, log_probs, returns, advantage):
        """
        Iterator to generate mini-batches of training data for the Proximal Policy Optimization (PPO) algorithm

            Paramerer:
                mini_batch_size (int): size of each mini batch
                states: array containing the states encountered during interactions.
                actions:  array containing the actions taken in response to the states
                log_probs: array containing the logarithm of the probability of taking the actions under the current policy
                returns: array containing the estimated returns (GAE values) for the actions taken
                advantage: array containing the advantages of taking the actions, which guide the policy update

            Return:
                states[rand_ids, :]: A mini-batch of states.
                actions[rand_ids, :]: A mini-batch of corresponding actions.
                log_probs[rand_ids, :]: A mini-batch of log probabilities associated with the selected actions.
                returns[rand_ids, :]: A mini-batch of estimated returns.
                advantage[rand_ids, :]: A mini-batch of advantages.

        Author: Luke Voss
        """
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def act(self, feature_vector):
        dist, self.value = self.model(feature_vector)

        if self.train:
            # Exploration: Sample from Action Distribution
            idx_action = dist.sample()
            self.action_logprob = dist.log_prob(idx_action)
        else:
            # Exploitation: Get Action with higest probability
            idx_action = dist.probs.argmax()  # TODO this correct?
        return ACTIONS[idx_action]

    def training_step(self, old_feature_state, new_feature_state, action_took: str, reward: float, is_terminal: bool):
        self.states.append(old_feature_state.to(self.device))
        idx_action = ACTIONS.index(action_took)
        self.actions.append(idx_action)
        self.rewards.append(reward)
        self.masks.append(1 - is_terminal)
        self.round_rewards += reward

        self.values.append(self.value)
        self.log_probs.append(self.action_logprob.unsqueeze(0))

        n_recorded_steps = len(self.actions)
        if (n_recorded_steps % UPDATE_PPO_AFTER_N_STEPS) == 0:
            if is_terminal:
                next_value = 0  # Next value doesn't exist
            else:
                _, next_value = self.model(new_feature_state)
            returns = self.compute_gae(next_value, self.rewards,
                                       self.masks, self.values)

            returns = torch.stack(returns).detach()
            log_probs = torch.stack(self.log_probs).detach()
            values = torch.stack(self.values).detach()
            states = torch.stack(self.states)
            actions = torch.tensor(self.actions, device=self.device).unsqueeze(1)
            advantages = returns - values

            # Update step of PPO algorithm
            if states.size(0) > 0:
                loss = self.update(PPO_EPOCHS_PER_EVALUATION, MINI_BATCH_SIZE,
                                   states, actions, log_probs, returns, advantages)
                self.loss_sum += loss
                self.n_updates += 1

            if is_terminal:
                print(' Total rewards of {}, Loss: {}'.format(
                    self.round_rewards, self.loss_sum/self.n_updates))
                self.round_rewards = 0

            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.masks = []
            self.log_probs = []

            #return loss

    def save_model(self, model_name="ppo_model"):
        model_path = os.path.join("./models", model_name + ".pt")
        torch.save(self.model.state_dict(), model_path)
