from ast import Raise


class Agent(object):
    '''
    Abstract Delegate for an agent.
    No implementation in Agent.
    
    '''
    def __init__():
        pass

    def choose_action(self):
        '''
        Please override the virtual fucntion to define the real behavior for each world tick.
        '''
        raise NotImplementedError()

    def destroy(self):
        '''
        Please override the virtual function to destroy the agent in Carla world.
        '''
        raise NotImplementedError()