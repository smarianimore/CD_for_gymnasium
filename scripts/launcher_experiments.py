import gymnasium as gym
import multiprocessing
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from scripts.utils import wrappers

LABEL_FROZEN_LAKE = 'FrozenLake-v1'
LABEL_TAXI = 'Taxi-v3'

ENVIRONMENTS = ['FrozenLake-v1', 'Taxi-v3']
ALGORITHMS = [DQN, PPO]
TRAINING_EPISODES = 10000
EVALUATION_EPISODES = 10
N_ENVIRONMENTS = 10


def make_vectorized_env(env_id, num_envs):
    def make_env():
        return env_wrapper(gym.make(env_id))

    return make_vec_env(make_env, n_envs=num_envs)


def train_and_evaluate(env_name, algorithm_class):
    env = make_vectorized_env(env_name, num_envs=N_ENVIRONMENTS)

    model = algorithm_class("MultiInputPolicy", env)
    model.learn(total_timesteps=TRAINING_EPISODES, progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVALUATION_EPISODES)

    result = f"{algorithm_class} on {env_name} - Mean Reward: {mean_reward} +/- {std_reward}"

    env.close()

    return result


def env_wrapper(environment):
    env_name = environment.spec.id
    if env_name == LABEL_FROZEN_LAKE:
        environment = wrappers.FrozenLake_ObservationSpaceWrapper(environment, (3, 3))
    elif env_name == LABEL_TAXI:
        environment = wrappers.Taxi_ObservationSpaceWrapper(environment, (3, 3))
    else:
        print(f'Wrapper for {env_name} is not still available')

    return environment


def main():
    # Create a pool of processes. The Number of processes by default is set to the number of CPUs available.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Create a list of all combinations of environments and algorithms
        tasks = [(env, alg) for env in ENVIRONMENTS for alg in ALGORITHMS]

        # Map train_and_evaluate function to all combinations
        results = pool.starmap(train_and_evaluate, tasks)

        for result in results:
            print(result)


if __name__ == '__main__':
    main()
