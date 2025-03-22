import curses
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from envs import MontezumaInfoWrapper, MaxAndSkipEnv
import gym
from enum import IntEnum

class actions(IntEnum):
    INVALID         = -1
    NOOP            = 0
    FIRE            = 1
    UP              = 2
    RIGHT           = 3
    LEFT            = 4
    DOWN            = 5
    UP_RIGHT        = 6
    UP_LEFT         = 7
    DOWN_RIGHT      = 8
    DOWN_LEFT       = 9
    UP_FIRE         = 10
    RIGHT_FIRE      = 11
    LEFT_FIRE       = 12
    DOWN_FIRE       = 13
    UP_RIGHT_FIRE   = 14
    UP_LEFT_FIRE    = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE  = 17

class MonteDataCollector:
    def __init__(self):
        self.save_dir = 'resources/monte_state/'
        
        self.env = MontezumaInfoWrapper(
            MaxAndSkipEnv(gym.make("MontezumaRevengeNoFrameskip-v4",
                                    render_mode="rgb_array"), False)
            , 1)
        self.env.reset()

        # init visualization
        self.fig = plt.figure(num=1, figsize=(10,10), clear=True)
        self.ax = self.fig.add_subplot()
        screen = self.env.render()
        self.ax.clear()  # Clear the axes
        self.ax.imshow(screen)  # Update the image
        plt.show(block=False)


    def visualize_env(self, pause=0.01):
        # update env visualization for current state
        screen = self.env.render()
        self.ax.clear()
        self.ax.imshow(screen)
        plt.draw()
        plt.pause(pause)


    def set_state(self, state_file):
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        
        self.env.unwrapped.ale.restoreState(state)
        self.perform_action(actions.NOOP, 1)
        
        self.visualize_env()
    
    def _create_open(self, filename, mode):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return open(filename, mode)

    def save_data(self, file_name, state):   
        with self._create_open(file_name, 'wb') as f:
            pickle.dump(state, f)

    def perform_action(self, action, steps):

        for _ in range(steps):
            obs, _, _, _ = self.env.step(action)
            self.visualize_env()            

    def perform_action_with_delay(self, key):
            action_map = {
                ord('w'): actions.UP,
                ord('s'): actions.DOWN,
                ord('a'): actions.LEFT,
                ord('d'): actions.RIGHT,
                ord('q'): actions.UP_LEFT,
                ord('e'): actions.UP_RIGHT,
                ord('z'): actions.DOWN_LEFT,
                ord('c'): actions.DOWN_RIGHT,
                curses.KEY_UP: actions.UP_FIRE,
                curses.KEY_DOWN: actions.DOWN_FIRE,
                curses.KEY_LEFT: actions.LEFT_FIRE,
                curses.KEY_RIGHT: actions.RIGHT_FIRE,
                ord('1'): actions.FIRE,
                ord(' '): actions.NOOP,
            }

            if key in action_map:
                action = action_map[key]
                self.perform_action(action, 1)
            self.visualize_env()


    def run(self):
        curses.wrapper(self.collect_data)


    def collect_data(self, stdscr):
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.clear()
        stdscr.scrollok(True)
        stdscr.addstr("Now collecting data! Press ESC to exit.\n")
        stdscr.addstr("Press W, A, S, D to move. Q, E, Z, C for combined movements.\n")
        stdscr.addstr("Use arrow keys for jumping movements.\n")
        stdscr.addstr("Press B to save data.\n")

        self.visualize_env()

        while True:
            key = stdscr.getch()  # Get a single key press
            stdscr.addstr(f"pressed {chr(key)} \n")
            
            if key == 27:  # ESC key to exit
                break
                

            elif key == ord('b'):
                stdscr.addstr(f"Save state? (y/n)\n")
                key = stdscr.getch()
                
                if key == ord('y'):
                    # ask for save file name prefix
                    stdscr.addstr(f"Enter save file name \n")
                    prefix = stdscr.getstr().decode('utf-8')
                    stdscr.addstr(f"Saving data to {self.save_dir+prefix}\n")
                    self.save_data(os.path.join(self.save_dir, prefix), self.env.unwrapped.ale.cloneState())
                    stdscr.addstr("Data saved!\n")
                else:
                    stdscr.addstr("Data not saved. Still collecting data.\n")
                
            elif key in (ord('w'), ord('s'), ord('a'), ord('d'), ord('q'), ord('e'), ord('z'), ord('c'),
                         curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT, ord('1'), ord(' ')):
                self.perform_action_with_delay(key)
                
            stdscr.refresh()

if __name__ == "__main__":
    collector = MonteDataCollector()
    
    collector.set_state('resources/monte_state/room_5/top_ladder_key.pkl')
    
    collector.run()