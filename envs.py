import gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        
        terminated = False
        truncated = False
        
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        info['terminated'] = terminated
        info['truncated'] = truncated

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))
        info['ram'] = unwrap(self.env).ale.getRAM()
        
        info = self.get_current_info(info)

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        obs, info = self.env.reset()
        return obs
    
    def get_current_info(self, info):
        ram = info['ram']
    
        info["lives"] = self.get_num_lives(ram)
        info["player_x"] = self.get_player_x(ram)
        info["player_y"] = self.get_player_y(ram)
        info["room_number"] = self.get_current_room()
        info["jumping"] = self.get_is_jumping(ram)
        info["dead"] = self.get_is_player_dead(ram)
        info["falling"] = self.get_is_falling(ram)
        info["uncontrollable"] = self.get_is_in_non_controllable_state(ram)
        info["left_door_open"] = self.get_is_left_door_unlocked(ram)
        info["right_door_open"] = self.get_is_right_door_unlocked(ram)
        info["inventory"] = self.get_player_inventory(ram)

        return info
    
    def get_current_position(self):
        ram = self.get_current_ram()
        return self.get_player_x(ram), self.get_player_y(ram)

    def get_player_x(self, ram):
        return int(self.getByte(ram, 'aa'))

    def get_player_y(self, ram):
        return int(self.getByte(ram, 'ab'))

    def get_num_lives(self, ram):
        return int(self.getByte(ram, 'ba'))

    def get_player_inventory(self, ram):
        # 'torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer'
        return format(self.getByte(ram, 'c1'), '08b')
    
    def get_is_falling(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_is_jumping(self, ram):
        return int(self.getByte(ram, 'd6') != 0xFF)

    def get_room_number(self, ram):
        return int(self.getByte(ram, '83'))
    
    def get_player_status(self, ram):
        status = self.getByte(ram, '9e')
        status_codes = {
            0x00: 'standing',
            0x2A: 'running',
            0x3E: 'on-ladder',
            0x52: 'climbing-ladder',
            0x7B: 'on-rope',
            0x90: 'climbing-rope',
            0xA5: 'mid-air',
            0xBA: 'dead',  # dive 1
            0xC9: 'dead',  # dive 2
            0xC8: 'dead',  # dissolve 1
            0xDD: 'dead',  # dissolve 2
            0xFD: 'dead',  # smoke 1
            0xE7: 'dead',  # smoke 2
        }
        return status_codes[status]

    def get_is_player_dead(self, ram):
        player_status = self.get_player_status(ram)
        dead = player_status == "dead"
        time_to_spawn = self.getByte(ram, "b7")
        respawning = time_to_spawn > 0
        return dead or respawning

    def get_is_in_non_controllable_state(self, ram):
        player_status = self.get_player_status(ram)
        return self.get_is_jumping(ram) or \
            player_status in ("mid-air") or\
            self.get_is_falling(ram) or \
            self.get_is_player_dead(ram)

    def get_is_left_door_unlocked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        left_door = objects[0]
        locked = int(left_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
        return not locked

    def get_is_right_door_unlocked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        right_door = objects[1]
        locked = int(right_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
        return not locked
    
    @staticmethod
    def _getIndex(address):
        assert type(address) == str and len(address) == 2 
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row*16+col
    
    @staticmethod
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = MontezumaInfoWrapper._getIndex(address)
        return ram[idx]


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_id:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, info = self.env.step(action)

            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                    info.get('episode', {}).get('visited_rooms', {})))

                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward, info])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)
