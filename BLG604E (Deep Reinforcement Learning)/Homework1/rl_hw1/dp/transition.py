""" Produce the necessary transition map for dynamic programming.
    Author: Anıl Öztürk / 504181504
"""
from collections import defaultdict
import numpy as np


def make_transition_map(initial_board):
    """ Return the transtion map for passable(other than wall cells) states.
    In a state S an action A is chosen, there are four possibilities:
    - Intended action can be picked
    - 3 of the remaning action can be picked by the environment.
    Structure of the map:

    map[S][A] -> [
        (p_0, n_s, r_0, t_0), # Quad tuple of transition for the action 0
        (p_1, n_s, r_1, t_1), # Quad tuple of transition for the action 1
        (p_2, n_s, r_2, t_2), # Quad tuple of transition for the action 2
        (p_3, n_s, r_3, t_3), # Quad tuple of transition for the action 3
    ]

    p_x denotes the probability of transition by action "x"
    r_x denotes the reward obtained during the transition by "x"
    t_x denotes the termination condition at the new state(next state)
    n_s denotes the next state

    S denotes the space of all the non-wall states
    A denotes the action space which is range(4)
    So each value in map[S][A] is a length 4 list of quad tuples.


    Arguments:
        - initial_board: Board of the Mazeworld at initialization

    Return:
        transition map
    """
    
    grid = np.asarray(initial_board)
    heigth, width = grid.shape

    nactions = 4
    nstates = heigth * width

    target_state = np.argwhere(grid == 64)[0] ## Getting the target state

    # O: up
    # 1: down
    # 2: left
    # 3: right


    all_states = np.argwhere(grid != 35) ## selecting all non-wall states
    transition_map = {tuple(state): {act: [] for act in range(nactions)} for state in all_states} ## creating a dict with (s,a) keys

    def is_acceptable_state(row , arr):
        return (arr == row).all(axis=1).any()

    for xdim,ydim in all_states:   

        for action in range(nactions):

            transition_map[xdim,ydim][action] = []

            for stochastic_action in range(nactions):

                res_x = xdim
                res_y = ydim
                
                if action == stochastic_action:
                    prob = 0.7
                else:
                    prob = 0.1
                
                if stochastic_action==0 and is_acceptable_state([res_x-1, res_y], all_states):
                    res_x -= 1
                elif stochastic_action==1 and is_acceptable_state([res_x+1, res_y], all_states):
                    res_x += 1

                if stochastic_action==2 and is_acceptable_state([res_x, res_y-1], all_states):
                    res_y -= 1
                elif stochastic_action==3 and is_acceptable_state([res_x, res_y+1], all_states):
                    res_y += 1

                if res_x == target_state[0] and res_y == target_state[1]:
                    reward = 1.0
                    finished = True
                
                else:
                    reward = 0.
                    finished = False

                transition_map[xdim,ydim][action].append( (prob, (res_x,res_y), reward, finished) )

    return transition_map


