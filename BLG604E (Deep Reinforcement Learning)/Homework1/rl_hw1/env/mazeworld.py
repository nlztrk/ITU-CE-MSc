""" Very simple maze environment
    Author: Tolga Ok & Nazim Kemal Ure
"""
import gym
from gym import spaces
import numpy as np
import random
import pycolab
import time
from pycolab.prefab_parts import sprites
from pycolab.ascii_art import Partial
from pycolab.cropping import ObservationCropper, ScrollingCropper

from .colabenv import ColabEnv

WORLDMAP = ["##########",
            "#     #  #",
            "# ### #  #",
            "#   #    #",
            "### ### ##",
            "# #  #   #",
            "#    ##  #",
            "### ##  ##",
            "#P   # @ #",
            "##########"]
ENV_LENGTH = 2000


class PlayerSprite(sprites.MazeWalker):
    """ Sprite of the agent that can move to 3 different directions.
    """

    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable="#")

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, layers  # Unused

        if actions == 0:    # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)
        elif actions == 4:  # do nothing?
            self._stay(board, the_plot)


class WallDrape(pycolab.things.Drape):

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del self, actions, board, layers, backdrop, things, the_plot


class CashDrape(pycolab.things.Drape):
    """A `Drape` handling all of the coins.

    This Drape detects when a player traverses a coin, removing the coin and
    crediting the player for the collection. Terminates if all coins are gone.
    It also terminates if the maximum number of transitions is reached.
    """

    def __init__(self, curtain, character):
        super().__init__(curtain, character)
        self.env_length = ENV_LENGTH

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_pattern_position = things['P'].position

        if self.env_length <= 0:
            self.env_length = ENV_LENGTH
            the_plot.terminate_episode()
        else:
            if self.curtain[player_pattern_position]:
                the_plot.add_reward(1)
                self.curtain[player_pattern_position] = False
                if not self.curtain.any():
                    the_plot.terminate_episode()
            else:
                the_plot.add_reward(0)
                self.env_length -= 1


class MazeWorld(ColabEnv):
    """ Simple grid world where there are 3 different possible drapes and an
    agent. Agent is allowed to move into empty cells as well as into goal goal
    states but not to the wall drapes. When all the coins are collected or 200
    timesteps is past the environment terminates.
    Arguments:
        - size: Length of the edge of the world in turms of cell
        - cell_size: Size of each cell for renderer.
        - colors: Color dictionary that maps envionment characters to colors
        - render_croppers: Croppers for the renderer. Renderer initialize cells
            for each cropper. These croppers do not make any difference in the
            environment mechanics.
    """
    COLORS = {
        "P": "#00B8FA",
        " ": "#DADADA",
        "@": "#DADA22",
        "#": "#989898"
    }

    def __init__(self, cell_size=50, colors=None,
                 render_croppers=None, worldmap=None):
        self.world_map = worldmap or WORLDMAP
        super().__init__(cell_size=cell_size,
                         colors=colors or self.COLORS,
                         render_croppers=render_croppers)
        self.to_feature = self._cartesian

    def _cartesian(self, observation):
        return tuple(np.argwhere(observation.board == 80)[0])

    def _init_game(self):
        game = pycolab.ascii_art.ascii_art_to_game(
            art=self.world_map,
            what_lies_beneath=" ",
            sprites={"P": Partial(PlayerSprite)},
            drapes={"#": Partial(WallDrape),
                    "@": Partial(CashDrape)},
            update_schedule=[["P"], ["#"], ["@"]],
            z_order="#P@"
        )
        return game


class StochasticMaze(MazeWorld):
    """ Stochastic version of the MazeWorld where the intended action is
    choosen with 70% probability while remaminings are choosen with 10%
    probability each.
    """

    def step(self, action):
        if random.uniform(0, 1) > 0.7:
            action = random.choice(list(set((0, 1, 2, 3)) - {action}))
        return super().step(action)


# Example usage
if __name__ == "__main__":
    croppers = [
        ScrollingCropper(rows=5, cols=5, to_track=['P'],
                         initial_offset=(8, 1), pad_char="#", scroll_margins=(2, 2)),
        ObservationCropper()
    ]
    env = MazeWorld(render_croppers=croppers)
    for i in range(100):
        done = False
        state = env.reset()
        print(state.shape)
        while not done:
            action = random.randint(0, 4)
            state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.1)
            print(reward)
            if done:
                print("Done")
                break
