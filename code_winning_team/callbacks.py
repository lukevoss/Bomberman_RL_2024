
import pickle
import random

from agent_code.q_agent_v2.train import *




cwd = os.path.abspath(os.path.dirname(__file__))


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not self.train:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: Game) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    state = state_to_features(game_state)
    state = tuple(state)
    
    if  self.model.get(state): # If the state has been already ecountered
        model_result = self.model[state]
        if model_result.values() is not None: # If the encountered state has values in it for the actions
             
            random_int = random.uniform(0,1)
            if self.train:# If in training MODE
                epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * game_state['round'])
                self.logger.debug(f"Randomness is at {epsilon}")

                if random_int <= epsilon: # Go for random action for a decreasing percentage of randomness
                    action = random.choice(ACTIONS)# without bomb
                    self.logger.debug(f"***************Choosing  {action}  purely at random in {state} ")
                    return action
                # Choose from the Q_Table the best action that is stored, if more than one best action, choose one randomly
                max_value = max(model_result.values())
                possible_actions = [key for key, val in model_result.items() if abs(max_value - val) < 1e-4]
                action = random.choice(possible_actions)
                self.logger.info(f"Picking {action} from state {state}.")
                self.logger.info(f"Number of encountered states: {len(self.model)}  in round: {game_state['round']}.")
                return action
            # Having some randomness even if there is best action 
            # epsilon =  MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * game_state['step']) #
            epsilon = .01
            if random_int <= epsilon:
                action = random.choice(ACTIONS)
                self.logger.debug(f"*************Choosing {action} purely at random in {state}")
                return action
            # Choosing from Q_Table
            action = np.argmax(list(model_result.values()))
            action = ACTIONS[action]
            self.logger.info(f"Picking {action} from state {state}.")
            return action
    else:
        # Add the new state to the q_table
        action = random.choice(ACTIONS)
        self.model[state]= dict.fromkeys(ACTIONS, ZERO)
        # print(state)
        return action
        
