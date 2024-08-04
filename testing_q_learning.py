class GridWorld:
    def __init__(self, width, height, start_position, goal_position):
        self.width = width
        self.height = height
        self.current_position = start_position
        self.goal_position = goal_position
    
    def reset(self):
        self.current_position = (0, 0)
        return self.current_position

    def step(self, action):
        x, y = self.current_position
        if action == 0:  # up
            self.current_position = (max(x - 1, 0), y)
        elif action == 1:  # down
            self.current_position = (min(x + 1, self.height - 1), y)
        elif action == 2:  # left
            self.current_position = (x, max(y - 1, 0))
        elif action == 3:  # right
            self.current_position = (x, min(y + 1, self.width - 1))

        reward = -1  # default reward
        done = self.current_position == self.goal_position
        if done:
            reward = 100  # reward for reaching the goal

        return self.current_position, reward, done

    def get_possible_actions(self):
        return [0, 1, 2, 3]  # up, down, left, right