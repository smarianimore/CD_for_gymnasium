from typing import Tuple, Optional, Union, Dict
import numpy as np
from stable_baselines3 import DQN


class CustomDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(CustomDQN, self).__init__(*args, **kwargs)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        print(observation)
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state


def algo_wrappers_manager(algorithm_class, environment):
    if algorithm_class == DQN:
        algorithm = CustomDQN('MlpPolicy', environment)
    else:
        print(f'Wrapper for {algorithm_class} is not still available')
        algorithm = algorithm_class('MlpPolicy', environment)

    return algorithm
