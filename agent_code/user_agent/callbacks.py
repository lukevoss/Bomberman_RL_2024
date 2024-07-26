def setup(self):
    pass


def act(self, game_state: dict):
    print(game_state['bombs'])
    print(game_state['user_input'])

    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
