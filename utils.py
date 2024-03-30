import json
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.autoscale()
    plt.show(block=False)
    plt.pause(.1)
    
def normalize_feature(value, max_value):
    """Normalize a single feature."""
    return value / max_value

def normalize_game_state(state):
    max_score = 10000  # Example max score for normalization
    max_level = 20  # Example max level for normalization

    grid = np.array(state['grid'], dtype=np.float32).flatten()
    cur_shape = np.array(state['cur_shape'], dtype=np.float32).flatten()
    level = normalize_feature(state['level'], max_level)
    score = normalize_feature(state['score'], max_score)

    return np.concatenate((grid, cur_shape, [level, score]))

def normalize_game_state_tensor(state):
    normalized_state = normalize_game_state(state)

    # Convert to PyTorch tensor, adding an extra dimension to mimic batch size of 1
    return torch.tensor(normalized_state, dtype=torch.float).unsqueeze(0)

# Function to log game state and action to a file
def log_data(game_state, action, seq_id, file_path="data\\tetris_training_data.json"):
    log_entry = {
        "seq_id": seq_id,
        "state": game_state,
        "action": action
    }
    with open(file_path, "a") as file:
        json.dump(log_entry, file)
        file.write("\n")  # Write a newline for readability

