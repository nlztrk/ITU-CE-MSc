""" Run multiple environments in parallel.

    Wrap your environment with ParallelEnv to run in parallel. A wrapped
    parallel environment provides step and reset functions. After an initial
    reset call, there is no need for additional reset calls. If one of the
    environment terminates after the step call, it automatically resets the
    environment and sends initial observation as the next state. Thereby, users
    of this wrapper must be AWARE of the fact that in case of termination
    next state is the initial observation of the new episode.
"""
from torch.multiprocessing import Process, Pipe
import torch
from collections import namedtuple
import numpy as np


class ParallelEnv():
    """ Synchronize multi envirnment wrapper.

        Workers are communicated through pipes where each worker runs a single
        environment. Initiation is started by calling <start> method or using
        "with" statement. After initiating the workers, step function can be
        called indefinitely. In case of termination, each worker restarts it's
        own environment and returns the first state of the restarted
        environment instead of the last state of the terminated one.
            Arguments:
                - n_env: Number of environments
                - env_maker_fn: Function that returns an environment

        Example:
            >>> p_env = ParallelEnv(n, lambda: gym.make(env_name))
            >>> states = p_env.reset()
            >>>     actions = policy(states)
            >>>     for i in range(TIMESTEPS):
            >>>         states, rewards, dones = p_env.step(actions)
    """

    EnvProcess = namedtuple("EnvProcess", "process, remote")

    def __init__(self, n_envs, env_maker_fn):
        self.env_maker_fn = env_maker_fn
        self.n_envs = n_envs
        self.started = False

    def reset(self):
        """ Initiate the worker processes and return all the initial states.
            Return:
                - state: The first observations as a stacked array
            Raise:
                - RuntimeError: If called twice without close
        """
        if self.started is True:
            raise RuntimeError("cannot restart without closing")

        env_processes = []
        for p_r, w_r in (Pipe() for i in range(self.n_envs)):
            process = Process(target=self.worker,
                              args=(w_r, self.env_maker_fn),
                              daemon=True)
            env_processes.append(self.EnvProcess(process, p_r))
            process.start()
            p_r.send("start")
        self.env_processes = env_processes

        state = np.stack(remote.recv() for _, remote in self.env_processes)
        self.started = True
        return state

    def step(self, actions):
        """ Steps all the workers(environments) and return stacked
        observations, rewards, and termination arrays. When a termination
        happens in one of the workers, it returns the first observation of the
        restarted environment instead of returning the next-state of the
        terminated episode.
            Arguments:
                - actions: Stacked array of actions. Dim: (#env, #act)
            Return:
                - Stacked state, reward and done arrays. Dimension of state:
                    (#env, #obs), reward: (#env, 1), done: (#env, 1)
            Raise:
                - RuntimeError: If called before start
                - ValueError: If argument <actions> is not a 2D array
                - ValueError: If #actions(0th dimension) is not equal to
                    #environments
        """
        if self.started is False:
            raise RuntimeError("call <start> function first!")
        if len(actions.shape) != 2:
            raise ValueError("<actions> must be 2 dimensional!")
        if actions.shape[0] != self.n_envs:
            raise ValueError("not enough actions!")
        actions = actions.squeeze(-1)
        for act, (_, remote) in zip(actions, self.env_processes):
            remote.send(act)

        state, reward, done = [np.stack(batch) for batch in zip(*(
            remote.recv() for _, remote in self.env_processes))]
        return (state,
                reward.reshape(-1, 1).astype(np.float32),
                done.reshape(-1, 1).astype(np.float32))

    def close(self):
        """ Terminate and join all the workers.
        """
        for process, remote in self.env_processes:
            remote.send("end")
            process.terminate()
            process.join()
        self.started = False

    def __enter__(self):
        """ Called when used with python's "with" statement """
        return self.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Called when closing the python's "with" statement """
        self.close()

    @staticmethod
    def worker(remote, env_maker_fn):
        """ Start when the initial start signal is received from <reset>
        call. Following the start signal, the first observation array is
        sent through the pipe. Then, the worker waits for the action from the
        pipe. If the action is "end" string, then break the loop and terminate.
        Otherwise, the worker steps the environment and sends (state, reward,
        done) array triplet.

        Arguments:
            - remote: Child pipe
            - env_maker_fn: Function that return an env object
        """
        env = env_maker_fn()
        state = env.reset()
        # Wait for the start command
        remote.recv()
        remote.send(state)
        while True:
            action = remote.recv()
            if action == "end":
                break
            state, reward, done, info = env.step(action.item())
            if done:
                state = env.reset()
            remote.send((state, reward, done))
