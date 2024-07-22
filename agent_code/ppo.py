

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
        delta = rewards[step] + gamma * \
            values[step + 1] * masks[step] - values[step]
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
