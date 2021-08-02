# import PyTorch libraries
import torch


# import gym and nes-py libraries
from gym.wrappers import FrameStack
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# import our custom helpers
import Utilities.helpers as helpers
import Utilities.Agent as Mario
import Utilities.Logger as log

from pathlib import Path
import datetime


game_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
game_env = JoypadSpace(game_env, COMPLEX_MOVEMENT)


# Applying Wrappers to the game environment
game_env = helpers.SkipFrame(game_env, num_skips=4)
game_env = helpers.ObserveInGrayscale(game_env)
game_env = helpers.ResizeObservation(game_env, shape=84)
game_env = FrameStack(game_env, num_stack=4)

# Check to see if GPU training is available
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario.Mario(state=(4, 84, 84), action=game_env.action_space.n, save_directory=save_dir)

logger = log.MetricLogger(save_dir)

episodes = 25

for e in range(episodes):

    state = game_env.reset()
    while True: # Loop until the episode is done
        action = mario.act(state)
        next_state, reward, done, info = game_env.step(action)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state

        # Check if episode is done
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 10 == 0:
        logger.record(episode=e,
                      epsilon=mario.exploration_rate,
                      step=mario.current_step)