import gymnasium as gym
import numpy as np

LABEL_GYM_OBS_PART = 'gym_obs'
LABEL_NEW_OBS_PART = 'new_obs'

LABEL_BOUNDARY = '*borders*'
LABEL_AGENT_WRAPPER = '*A'


def env_wrappers_manager(environment):
    """env_name = environment.spec.id
    if env_name == 'FrozenLake-v1':
        environment = FrozenLake_ObservationSpaceWrapper(environment, (3, 3))
    elif env_name == 'Taxi-v3':
        environment = Taxi_ObservationSpaceWrapper(environment, (3, 3))
    else:
        print(f'Wrapper for {env_name} is not still available')"""

    return environment


class FrozenLake_ObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, environment, observation_matrix_shape):
        super().__init__(environment)

        # This is the space for the typical observations from the environment
        self.gym_obs = environment.observation_space
        self.world = np.array(environment.get_wrapper_attr("desc").tolist(), dtype='<U10')

        self.world_rows, self.world_cols = self.world.shape

        self.observation_matrix_shape = observation_matrix_shape

        # Define a new observation space that includes both the typical and new observations
        self.observation_space = gym.spaces.Dict({
            LABEL_GYM_OBS_PART: self.gym_obs,
            LABEL_NEW_OBS_PART: gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_matrix_shape,
                                               dtype=np.float32)
        })

    def observation(self, obs):
        matrix = self._define_new_observations(obs)
        new_obs = matrix_assignment_numbers_to_strings(matrix)
        new_observation = {
            LABEL_GYM_OBS_PART: obs,
            LABEL_NEW_OBS_PART: new_obs
        }

        return new_observation

    def _define_new_observations(self, obs):

        def _replace_S_with_F(matrix):
            # Iterate through each row in the matrix
            for i in range(len(matrix)):
                # Iterate through each element in the row
                for j in range(len(matrix[i])):
                    # Check if the element is "S"
                    if matrix[i][j] == "S":
                        # Replace "S" with "F"
                        matrix[i][j] = "F"
            return matrix

        def _get_full_matrix(obs2):
            # Create a copy of the base grid for manipulation
            matrix = np.copy(self.world)
            # Convert scalar position index to 2D grid position
            row, col = divmod(obs2, self.world_rows)
            # We first replace the original start 'S' with 'F' since the agent has moved
            start_row, start_col = np.where(self.world == b'S')
            if len(start_row) > 0 and len(start_col) > 0:
                matrix[start_row[0], start_col[0]] = 'F'
            # Now mark the agent's current position with 'agent'
            matrix[row, col] = LABEL_AGENT_WRAPPER
            return matrix

        def get_obs_matrix(obs1):
            start_matrix = _get_full_matrix(obs1)

            row, col = divmod(obs1, self.world_rows)
            # Create a 3x3 submatrix centered on the agent
            submatrix = np.full(self.observation_matrix_shape, LABEL_BOUNDARY, dtype='<U10')
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sub_row = row + i
                    sub_col = col + j
                    if 0 <= sub_row < 4 and 0 <= sub_col < 4:
                        submatrix[i + 1, j + 1] = start_matrix[sub_row, sub_col]

            submatrix = _replace_S_with_F(submatrix)

            return submatrix.tolist()

        return get_obs_matrix(obs)


class Taxi_ObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, environment, observation_matrix_shape):
        super().__init__(environment)
        self.gym_obs = environment.observation_space
        self.world = np.array(environment.get_wrapper_attr('desc').tolist(), dtype='<U10')
        self.world_rows, self.world_cols = self.world.shape
        self.locs = environment.get_wrapper_attr('locs')

        self.observation_matrix_shape = observation_matrix_shape

        # Adjust shape according to your data structure; this is just an example
        self.observation_space = gym.spaces.Dict({
            LABEL_GYM_OBS_PART: self.gym_obs,
            LABEL_NEW_OBS_PART: gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_matrix_shape,
                                               dtype=np.float32)
        })

    def observation(self, obs):
        matrix = self._define_new_observations(obs)
        new_obs = matrix_assignment_numbers_to_strings(matrix)

        return {
            LABEL_GYM_OBS_PART: obs,
            LABEL_NEW_OBS_PART: new_obs
        }

    def _define_new_observations(self, obs):
        def custom_decode(i):
            out = []
            out.append(i % 4)  # dest_idx
            i = i // 4

            out.append(i % 5)  # pass_loc
            i = i // 5

            out.append(i % 5)  # taxi_col
            i = i // 5

            out.append(i)  # taxi_row
            assert 0 <= i < 5  # Ensure taxi_row is within the correct range

            x = list(reversed(out))  # This needs to reverse the list to match the order of encode inputs
            return x

        def _get_full_matrix(obs2):
            taxi_row, taxi_col, pass_idx, dest_idx = custom_decode(obs2)
            dynamic_world = np.copy(self.world)  # Copy the static world for modifications

            # Update taxi location with 'T'
            dynamic_world[taxi_row][taxi_col] = LABEL_AGENT_WRAPPER

            # Update passenger and destination markers
            for idx, loc in enumerate(self.locs):
                pr, pc = loc
                if idx == pass_idx:  # Passenger is here
                    dynamic_world[pr][pc] = 'P'
                if idx == dest_idx:  # Destination is here
                    dynamic_world[pr][pc] += 'D'  # Append 'D' to indicate destination
            return dynamic_world

        def get_obs_matrix(obs1):
            matrix = _get_full_matrix(obs1)
            taxi_row, taxi_col, _, _ = custom_decode(obs1)
            submatrix = np.full(self.observation_matrix_shape, ' ', dtype='<U10')
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sub_row = taxi_row + i
                    sub_col = taxi_col + j
                    if 0 <= sub_row < self.world_rows and 0 <= sub_col < self.world_cols:
                        submatrix[i + 1, j + 1] = matrix[sub_row][sub_col]
            return submatrix

        return get_obs_matrix(obs)


def matrix_assignment_numbers_to_strings(matrix):
    """
    Replace each unique string in the matrix with a unique number.

    Args:
    matrix (np.array): A 2D numpy array of strings.

    Returns:
    np.array: A 2D numpy array where each string is replaced by a unique number.
    """
    # Dictionary to map strings to numbers
    string_to_number = {}
    current_number = 1

    # Assign numbers to each unique string
    for row in matrix:
        for item in row:
            if item not in string_to_number:
                string_to_number[item] = current_number
                current_number += 1

    # Substitute strings in the matrix with their corresponding numbers
    number_matrix = np.vectorize(string_to_number.get)(matrix)

    return number_matrix
