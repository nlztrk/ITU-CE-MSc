""" Abstract class for Gym Colab environments. Include commom utilities.
    Taken from gymcolab: https://github.com/TolgaOk/gymcolab
    Author: Tolga Ok & Nazim Kemal Ure
"""
from itertools import chain
import numpy as np
import gym
from gym import spaces
from pycolab import cropping
from pycolab.rendering import ObservationToFeatureArray

from .notebook_renderer import CanvasRenderer


class ColabEnv(gym.Env):
    """ Base class for the pycolab gym environments.

        Environments can be build on top of this class given that the
        <_init_game> method is defined where the game is constructed. There
        are two sets of croppers one for renderer and the other for
        observation. Renderer croppers are only for visualization while
        observation cropper is what the environment returns after applying
        feature mapping. The default feature mapping is observation to 3D
        tensor where each frame is a mask for the corresponding character
        from the world map.
        Arguments:
            - render_croppers: List of croppers for visualization purposes
                only (default: No cropping)
            - observation_cropper: A cropper for cropping observation to
                transform the state (defualt: No cropping)
            - n_actions: Number of discrete actions (default: 5, for north,
                west, eath, south and no-ops)
            - (remaining kwargs): Renderer keyword arguments
    """
    metadata = {
        # Only GUI is available right now
        "render.modes": ["plot", "gui", "console"],
        # Only cropped-map is available right now
        "obs.modes": ["onehot", "index", "cropped-map"]
    }

    def __init__(self,
                 render_croppers=None,
                 observation_cropper=None,
                 n_actions=5,
                 cartesian=True,
                 **renderer_kwargs):

        if render_croppers is None:
            render_croppers = [cropping.ObservationCropper()]
        self.render_croppers = render_croppers

        if observation_cropper is None:
            observation_cropper = cropping.ObservationCropper()
        self.observation_cropper = observation_cropper

        # Initialize the game inorder to get characters of the game
        game = self._init_game()
        self.observation_cropper.set_engine(game)
        # chars = set(game.things.keys()).union(game._backdrop.palette)
        chars = sorted(set(game.things.keys()))

        # Observation space is a 3D space where the depth is the number of
        # unqiue characters in the game map where as spatial dimensions are
        # number of rows and columns of the observation croper
        # (default: whole map).
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(chars),
                   self.observation_cropper.rows,
                   self.observation_cropper.cols),
            dtype=np.float32
        )
        # Action space is a simple discrete space with 5 actions including
        # moving left, right, up, down, and no-ops
        self.action_space = spaces.Discrete(n_actions)
        self.to_feature = ObservationToFeatureArray(chars)
        # Define game atribute as None to check if reset is called before step
        self.game = None
        self.renderer_kwargs = renderer_kwargs


    def step(self, action):
        """ Iterate the environment for one time step. Gym step function.
        Arguments:
            - action: Discerte action as integer.
        Raise:
            - assertion, If the given argument <action> is not an integer
            - assertion, If the given argument <action> is out of range
            - assertion, If <step> function is called before initial <reset>
                call
            - assertion, If <step> function is called after the termination
        Return:
            - cropped and mapped observation
            - immidiate reward
            - termination
            - info dictionary
        """
        n_act = self.action_space.n
        assert isinstance(action, int), ("Parameter <action> must be integer. "
                                         "Try using <.item()> or squeeze down "
                                         "the action.")
        assert action in range(n_act), ("Parameter action is out of range. "
                                        "<action> must be in range of "
                                        "{}".format(n_act))
        assert self.game is not None, ("Game is not initialized"
                                       "Call reset function before step")
        assert self._done is False, ("Step can not be called after the game "
                                     "is terminated. Try calling reset first")

        observation, reward, discount = self.game.play(action)
        done = self.game.game_over
        self.observation = observation
        return self.observation_wrapper(observation), reward, done, {}

    def reset(self):
        """ Initialize the game and set croppers at every call. ALso at the
        fist call initialize renderer.
        Return:
            - cropped and mapped observation
        """
        if self.game is None:
            self._renderer = None
        self.game = self._init_game()
        observation, reward, discount = self.game.its_showtime()
        self._done = self.game.game_over

        self.observation_cropper.set_engine(self.game)
        for cropper in self.render_croppers:
            cropper.set_engine(self.game)

        self.observation = observation
        return self.observation_wrapper(observation)

    def observation_wrapper(self, observation):
        """ Crop and map the observation using observation mapping function.
        """
        observation = self.observation_cropper.crop(observation)
        return self.to_feature(observation)

    def render(self):
        """ Render the last observation using renderer croppers.
            Raise:
                assertion, If <reset> function is not called initially
        """
        assert self.game is not None, ("Game is not initialized"
                                       "Call reset function before step")
        if self._renderer is None:
            self._renderer = CanvasRenderer(
                croppers=self.render_croppers, **self.renderer_kwargs)
        self._renderer(self.observation)

    def init_render(self):
        self.reset()
        self.render()
        return self._renderer.canvas

    @property
    def board(self):
        try:
            return self.game._board.board
        except AttributeError:
            raise RuntimeError("Environment is not initiated. Call reset()")

    def _init_game(self):
        """ This function need to be overwritten from the child environment class
        """
        raise NotImplementedError

    
