import numpy as np
import torch
import torch.nn.functional as F

from typing import List

import events as e
from .callbacks import state_to_features, normalize_state, ACTIONS

SAVE_EVERY_N_EPOCHS = 100

# Events
CONSTANT_PENALTY = "Constant Penalty"
WON_GAME = "Won the game"

#Hyper parameters:
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
LR_ACTOR               = 3e-4
LR_CRITIC               = 3e-4
LR = 1e-6
NUM_STEPS        = 20
MINI_BATCH_SIZE  = 5
PPO_EPOCHS       = 4

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

        Return:  
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

# Proximal Policy Optimization Algorithm
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
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

def ppo_update(self, ppo_epochs, mini_batch_size, states, expert_actions, log_probs, returns, advantages, clip_param=0.2):
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
        mean_bc_loss = 0
        batch_size = states.size(0)
        n_updates = ppo_epochs*(batch_size // mini_batch_size)

        for state, expert_action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, expert_actions, log_probs, returns, advantages):
            dist, value = self.model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(expert_action)

            # Behavioral Cloning Loss (Supervised Loss)
            bc_loss = F.cross_entropy(dist.probs, expert_action.squeeze())
            mean_bc_loss += bc_loss

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = (0.5 * critic_loss + actor_loss - 0.001 * entropy) + bc_loss
            mean_loss += loss

            self.optimizer.zero_grad()
            # loss.backward()
            loss.backward()
            self.optimizer.step()
            ("PPO updated")
    if n_updates > 0:
        mean_loss = mean_loss/n_updates
        mean_bc_loss = mean_bc_loss/n_updates
    return mean_loss, mean_bc_loss

def setup_training(self):
    """
    Initialise self for training purpose. 
    This function is called after `setup` in callbacks.py.

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.

    Author: Luke Voss
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.states = []
    self.expert_actions = []
    self.rewards = []
    self.values = []
    self.masks = []
    self.log_probs = []
    self.loss_sum = 0
    self.bc_loss_sum = 0
    self.n_updates = 0

    self.round_rewards = 0
    

    self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=LR)


def game_events_occurred(self, old_game_state: dict, expert_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    Keeps track of transitions and updates the Agent network

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.
            old_game_state (dict): The state that was passed to the last call of `act`.
            self_action (str): The action that you took.
            new_game_state (dict): The state the agent is in now.
            events (list): The events that occurred when going from  `old_game_state` to `new_game_state`

    Author: Luke Voss
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    events.append(CONSTANT_PENALTY)
    # TODO hand out own rewards and add to rewards list

    # Handle None actions
    if expert_action is None:
        expert_action = 'WAIT'

    # Normalize game states
    self.action_map, self.reverse_action_map = normalize_state(old_game_state)
    _,_ = normalize_state(new_game_state)
    normalized_expert_action = self.reverse_action_map(expert_action)
    idx_normalized_expert_action = torch.tensor([ACTIONS.index(normalized_expert_action)],device=self.device)
    old_feature_state = state_to_features(old_game_state).to(self.device)

    dist, value = self.model(old_feature_state)
    log_prob = dist.log_prob(idx_normalized_expert_action)

    done = 0

    reward = reward_from_events(self, events)
    self.round_rewards += reward
    # Save Tansistions in Lists
    self.states.append(old_feature_state)
    self.expert_actions.append(idx_normalized_expert_action)
    self.rewards.append(reward)
    self.values.append(value)
    self.masks.append(1 - done)
    self.log_probs.append(log_prob)
    

    if (new_game_state['step'] % TRANSITION_HISTORY_SIZE) == 0:
        next_state = state_to_features(new_game_state).to(self.device)
        _, next_value = self.model(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.stack(returns).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values    = torch.stack(self.values).detach()
        states    = torch.stack(self.states)
        expert_actions   = torch.stack(self.expert_actions)
        advantages = returns - values

        # Update step of PPO algorithm + Behavioral cloning of expert
        
        loss, bc_loss = ppo_update(self, PPO_EPOCHS, MINI_BATCH_SIZE, states, expert_actions, log_probs, returns, advantages)
        self.loss_sum += loss
        self.bc_loss_sum += bc_loss
        self.n_updates += 1
        
        
        
        self.states = []
        self.expert_actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []
        
        



def end_of_round(self, last_game_state: dict, last_expert_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.
            last_game_state (dict): The last state that was passed to the last call of `act`.
            last_action (str): The last action that you took.
            events (list): The last events that occurred in the last step

    Author: Luke Voss
    """
    events.append(CONSTANT_PENALTY)

    # Check if agent is winner of game 
    score_self = last_game_state['self'][1]
    score_others = [other[1] for other in last_game_state['others']]
    if all(score_self > score for score in score_others):
        events.append(WON_GAME)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    n_round = last_game_state['round']

    # Handle None Actions
    if last_expert_action is None:
        last_expert_action = 'WAIT'

    # Normalize states and actions
    self.action_map, self.reverse_action_map = normalize_state(last_game_state)
    normalized_expert_action = self.reverse_action_map(last_expert_action)
    idx_normalized_expert_action = torch.tensor([ACTIONS.index(normalized_expert_action)],device=self.device)
    last_feature_state = state_to_features(last_game_state).to(self.device)

    dist, value = self.model(last_feature_state)
    log_prob = dist.log_prob(idx_normalized_expert_action)

    reward = reward_from_events(self, events)
    self.round_rewards += reward
    done = 1

    # Save Tansistions in Lists
    self.states.append(last_feature_state)
    self.expert_actions.append(idx_normalized_expert_action)
    self.rewards.append(reward)
    self.values.append(value)
    self.masks.append(1 - done)
    self.log_probs.append(log_prob)
    
    next_value = 0 # Next value doesn't exist
    returns = compute_gae(next_value, self.rewards, self.masks, self.values)

    returns   = torch.stack(returns).detach()
    log_probs = torch.stack(self.log_probs).detach()
    values    = torch.stack(self.values).detach()
    states    = torch.stack(self.states)
    actions   = torch.stack(self.expert_actions)
    advantages = returns - values

    # Update step of PPO algorithm
    if states.size(0) > 0:
        loss, bc_loss = ppo_update(self, PPO_EPOCHS, MINI_BATCH_SIZE, states, actions, log_probs, returns, advantages)
        self.loss_sum += loss
        self.bc_loss_sum += bc_loss
        self.n_updates += 1
    
    print(' Total rewards of {}, Loss: {}, BC Loss: {}'.format(self.round_rewards,self.loss_sum/self.n_updates, self.bc_loss_sum/self.n_updates))

    self.states = []
    self.expert_actions = []
    self.rewards = []
    self.values = []
    self.masks = []
    self.log_probs = []
    self.loss_sum = 0
    self.bc_loss_sum = 0
    self.n_updates = 0
    self.round_rewards = 0

    # Store the model
    if (n_round % SAVE_EVERY_N_EPOCHS) == 0:
        model_path = "./models/imitation_model.pt"
        torch.save(self.model.state_dict(), model_path)


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the Rewards sum from all current events in the game

        Parameter:
            events (list[str]) = List of occured events

        Return:
            reward_sum [float] = Sum of all reward for occured events

    Author: Luke Voss
    """

    game_rewards = {
        e.COIN_COLLECTED: 0.5,
        e.INVALID_ACTION: -0.01,
        e.KILLED_SELF: 0.1, # It is better to kill himself than to get killed
        e.GOT_KILLED: -1,
        e.CRATE_DESTROYED: 0.05,
        e.KILLED_OPPONENT: 1,
        CONSTANT_PENALTY : -0.001,
        WON_GAME: 1
    }


    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
