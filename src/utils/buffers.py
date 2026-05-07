import numpy as np

"""
Utilities for saving and loading replay buffer states and actions.
"""

def save_replay_buffer_states_actions(buffer, path):
    states = buffer.states[:buffer.current_size]
    actions = buffer.actions[:buffer.current_size]

    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        size=buffer.current_size,
    )

def load_minibatches(path):
    minibatches = []
    with open(path, "rb") as f:
        while True:
            try:
                minibatch = np.load(f)
                minibatches.append(minibatch)
            except EOFError:
                break
    return np.array(minibatches)


def load_state_action_matrix(path):
    data = np.load(path)

    states = data["states"]
    actions = data["actions"]

    actions = actions.reshape(-1, 1)

    return np.hstack([states, actions])
