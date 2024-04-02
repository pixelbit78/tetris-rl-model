from agent import TetrisAgent
from tetris_game import TetrisGame
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_rewards(show_result=False, totals=[]):
    interval = 10
    plt.figure(1)
    rewards_t = np.array(totals)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

plt.ion()

game = TetrisGame()
agent = TetrisAgent(
    n_observations=len(game.restart_game()),
    n_actions=game.nb_actions,
    env_step_func=game.train_step,
    env_reset_func=game.restart_game,
    env_next_steps_func=game.get_next_states,
    env_score_func=game.get_score,
    plot_rewards_func=plot_rewards
)

agent.train()

plt.ioff()
plt.show()