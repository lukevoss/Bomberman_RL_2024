""" 
This File is called by the environment and manages the agents training
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach without feature engineering:
Board is representet in 15x15x7 vector
Symmetry of board is leveraged

Current status:
Agent learn, but gets stuck on bad local maxima. Behavioral cloning to solve issue, but results are still bad
Ideas:
Network not deep enough, reward system not dense enough, feature engeneering maybe nececarry


Author: Luke Voss
"""

import numpy as np
import torch

from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS

SAVE_EVERY_N_EPOCHS = 100
loop = 0

# Events
CONSTANT_PENALTY = "Constant Penalty"
WON_GAME = "Won the game"
BOMBED_1_TO_2_CRATES = "BOMBED_1_TO_2_CRATES"
BOMBED_3_TO_5_CRATES = "BOMBED_3_TO_5_CRATES"
BOMBED_5_PLUS_CRATES = "BOMBED_5_PLUS_CRATES"
GET_IN_LOOP = "GET_IN_LOOP"
PLACEHOLDER_EVENT = "PLACEHOLDER"
ESCAPE = "ESCAPE"
NOT_ESCAPE = "NOT_ESCAPE"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
AWAY_FROM_CRATE = "AWAY_FROM_CRATE"
SURVIVED_STEP = "SURVIVED_STEP"
DESTROY_TARGET = "DESTROY_TARGET"
MISSED_TARGET = "MISSED_TARGET"
WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"
CLOSER_TO_PLAYERS = "CLOSER_TO_PLAYERS"
AWAY_FROM_PLAYERS = "AWAY_FROM_PLAYERS"

#Hyper parameters:
LR = 1e-4
NUM_STEPS        = 100
MINI_BATCH_SIZE  = 25
PPO_EPOCHS       = 8

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

def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
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
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = self.model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            mean_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if n_updates > 0:
            mean_loss = mean_loss/n_updates

    return mean_loss

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
    self.actions = []
    self.rewards = []
    self.values = []
    self.masks = []
    self.log_probs = []
    self.loss_sum = 0
    self.n_updates = 0

    self.round_rewards = 0
    
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=LR)
    # self.optimizer =  torch.optim.Adam([
    #                     {'params': self.model.actor.parameters(), 'lr': LR_ACTOR},
    #                     {'params': self.model.critic.parameters(), 'lr': LR_CRITIC}
    #                 ])


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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

    # Hand out self shaped events
    events = get_events(self, old_game_state, self_action, events)

   

    normalized_action = self.reverse_action_map(self_action)
    idx_normalized_action = torch.tensor([ACTIONS.index(normalized_action)],device=self.device)
    done = 0
    reward = reward_from_events(self, events)
    self.round_rewards += reward

    # Save Tansistions in Lists
    self.states.append(state_to_features(old_game_state).to(self.device))
    self.actions.append(idx_normalized_action)
    self.rewards.append(reward)
    self.values.append(self.value)
    self.masks.append(1 - done)
    self.log_probs.append(self.action_logprob.unsqueeze(0))
    
    
    if (len(self.actions) % NUM_STEPS) == 0:
        next_state = state_to_features(new_game_state).to(self.device)
        _, next_value = self.model(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.stack(returns).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values    = torch.stack(self.values).detach()
        states    = torch.stack(self.states)
        actions   = torch.stack(self.actions)
        advantages = returns - values

        # Update step of PPO algorithm
        loss = ppo_update(self, PPO_EPOCHS, MINI_BATCH_SIZE, states, actions, log_probs, returns, advantages)
        self.loss_sum += loss
        self.n_updates += 1

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []
        

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
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

    # Hand out self shaped events
    events = get_events(self,last_game_state,last_action,events)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    n_round = last_game_state['round']


    normalized_action = self.reverse_action_map(last_action)
    idx_normalized_action = torch.tensor([ACTIONS.index(normalized_action)],device=self.device)
    reward = reward_from_events(self, events)
    self.round_rewards += reward
    done = 1

    # Save Tansistions in Lists
    self.states.append(state_to_features(last_game_state).to(self.device))
    self.actions.append(idx_normalized_action)
    self.rewards.append(reward)
    self.values.append(self.value)
    self.masks.append(1-done)
    self.log_probs.append(self.action_logprob.unsqueeze(0))
    
    if (last_game_state['step'] % NUM_STEPS) == 0:
        next_value = 0 # Next value doesn't exist
        returns = compute_gae(next_value, self.rewards, self.masks, self.values)

        returns   = torch.stack(returns).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values    = torch.stack(self.values).detach()
        states    = torch.stack(self.states)
        actions   = torch.stack(self.actions)
        advantages = returns - values

        # Update step of PPO algorithm
        if states.size(0) > 0:
            loss = ppo_update(self, PPO_EPOCHS, MINI_BATCH_SIZE, states, actions, log_probs, returns, advantages)
            self.loss_sum += loss
            self.n_updates += 1
        
        print(' Total rewards of {}, Loss: {}'.format(self.round_rewards,self.loss_sum/self.n_updates))

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []

    self.round_rewards = 0

    # Store the model
    if (n_round % SAVE_EVERY_N_EPOCHS) == 0:
        model_path = "./models/ppo_model.pt"
        torch.save(self.model.state_dict(), model_path)


############# event shaping #####################


def check_position(x, y, arena, object):
    # Check whether the current position is of object type.
    if object == 'crate':
        return arena[x, y] == 1
    elif object == 'free':
        return arena[x, y] == 0
    elif object == 'wall':
        return arena[x, y] == -1

def march_forward(x, y, direction):
    # Forward in direction.
    if direction == 'LEFT':
        x -= 1
    elif direction == 'RIGHT':
        x += 1
    elif direction == 'UP':
        y -= 1
    elif direction == 'DOWN':
        y += 1
    return x, y

def danger(x_agent, y_agent, field, explosion_map,bombs):
    """
    Function checks if Agent is currently in danger

        Parameter: 
            x_agent(int): x position of agent
            y_agent(int): y position of agent
            field (np.array(width, height)): Current field as in game_state['field']
            bombs(list): list of (x,y) tuple of each bombs coordinates

        Returns:
            (bool): True if in danger 
    
    Author: Luke Voss
    """
    # TODO check if position in explosion for versatility
    if explosion_map[x_agent,y_agent] != 0:
        return True
    if not bombs: 
        return False
    
    for (x_bomb, y_bomb) in bombs:

        # Check if standing on it
        if x_bomb == x_agent and y_bomb == y_agent:
            return True 
        
        x_difference = x_bomb - x_agent
        y_difference = y_bomb - y_agent

        # Check if Agent in reach of bomb if same y position
        if abs(x_difference) <= 3 and y_bomb == y_agent:
            step = 1 if x_difference > 0 else -1

            # Check if wall inbetween
            for i in range(1, abs(x_difference) + 1):
                if field[x_agent + i * step][y_agent] == -1:
                    return False
            return True

        # Check if Agent in reach if same x position
        if abs(y_difference) <= 3 and x_bomb == x_agent:
            step = 1 if y_difference > 0 else -1

            # Check if wall inbetween
            for i in range(1, abs(y_difference) + 1):
                if field[x_agent][y_agent + i * step] == -1:
                    return False
            return True
        
    return False




def get_events(self, old_game_state, self_action, events_src)->list:
    """
    get events
    """
    #events = copy.deepcopy(events_src)
    events = events_src.copy()

    # Check if agent is winner of game 
    events.append(CONSTANT_PENALTY)

    


    field = old_game_state['field']
    x_agent, y_agent = old_game_state['self'][3]
    bombs = [xy for (xy, t) in old_game_state['bombs']]
    opponents = old_game_state['others']
    coins = old_game_state['coins']
    explosion_map = old_game_state['explosion_map']

    if opponents:
        score_self = old_game_state['self'][1]
        score_others = [other[1] for other in old_game_state['others']]
        if all(score_self > score for score in score_others):
            events.append(WON_GAME)


    is_in_danger = danger(x_agent, y_agent, field, explosion_map, bombs)

    # Check if in danger and reward if escaping
    if is_in_danger:
        if self_action == 'WAIT' or self_action =='BOMB':
            events.append(NOT_ESCAPE)
        else:
            x_new, y_new = march_forward(x_agent,y_agent,self_action)
            # Test if action was valid
            if field[x_new,y_new] == 0:
                for (x_bomb, y_bomb) in bombs:
                    # Check if distance increased
                    if ((abs(x_bomb - x_new) > abs(x_bomb - x_agent)) or 
                        ((y_bomb - y_new) > abs(y_bomb - y_agent))):
                        events.append(ESCAPE)
                    else:
                        events.append(NOT_ESCAPE)
            else: 
                events.append(NOT_ESCAPE)
    else:
        # Check if in loop
        self.loop_count = self.agent_coord_history.count((x_agent,y_agent))
        
        # If the agent gets caught in a loop, he will be punished.
        if self.loop_count > 2:
            events.append(GET_IN_LOOP)

        if self_action == 'WAIT':
            # Reward the agent if waiting is necessary.
            if (danger(x_agent+1,y_agent,field,explosion_map,bombs) or 
                danger(x_agent-1,y_agent,field,explosion_map,bombs) or
                danger(x_agent,y_agent+1,field,explosion_map,bombs) or 
                danger(x_agent,y_agent-1,field,explosion_map,bombs)):
                events.append(WAITED_NECESSARILY)
            else:
                events.append(WAITED_UNNECESSARILY)
        
        elif self_action != 'BOMB':
            x_new, y_new = march_forward(x_agent,y_agent,self_action)
            for opponent in opponents:
                x_opponent, y_opponent = opponent[3]

                # Check if distance decreased
                if ((abs(x_opponent - x_new) > abs(x_opponent - x_agent)) or 
                    ((y_opponent - y_new) > abs(y_opponent - y_agent))):
                    events.append(CLOSER_TO_PLAYERS)
                else:
                    events.append(AWAY_FROM_PLAYERS)

            for x_coin, y_coin in coins:
                # Check if distance decreased
                if ((abs(x_coin - x_new) > abs(x_coin - x_agent)) or 
                    ((y_coin - y_new) > abs(y_coin - y_agent))):
                    events.append(CLOSER_TO_COIN)
                else:
                    events.append(AWAY_FROM_COIN)

    return events

def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the Rewards sum from all current events in the game

        Parameter:
            events (list[str]) = List of occured events

        Return:
            reward_sum [float] = Sum of all reward for occured events

    Author: Luke Voss
    """
    # Base rewards:

    aggressive_action = 0.3
    coin_action = 0.2
    escape = 0.6
    waiting = 0.5

    game_rewards = {
        # SPECIAL EVENTS
        ESCAPE: escape,
        NOT_ESCAPE: -escape,
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -waiting,
        CLOSER_TO_PLAYERS: aggressive_action,
        AWAY_FROM_PLAYERS: -aggressive_action,
        CLOSER_TO_COIN: coin_action,
        AWAY_FROM_COIN: -coin_action,
        CONSTANT_PENALTY : -0.001,
        WON_GAME: 10,
        GET_IN_LOOP : -0.025 * self.loop_count,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -1,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,

        # kills
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10, # TODO: make killed self positiv since its better to kill himself, than to get killed
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
    }



    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

