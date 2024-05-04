import gymnasium as gym
import multiprocessing
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from scripts.utils.env_wrappers import env_wrappers_manager
from scripts.utils.algo_wrappers import algo_wrappers_manager

ENVIRONMENTS = ['FrozenLake-v1', 'Taxi-v3']
ALGORITHMS = [DQN, PPO]
TRAINING_EPISODES = 10000
EVALUATION_EPISODES = 10
N_ENVIRONMENTS = 10


def make_vectorized_env(env_id, num_envs):
    def make_env():
        return env_wrappers_manager(gym.make(env_id))

    return make_vec_env(make_env, n_envs=num_envs)


def train_and_evaluate(env_name, algorithm_class):
    env = make_vectorized_env(env_name, num_envs=N_ENVIRONMENTS)

    model = algo_wrappers_manager(algorithm_class, env)
    model.learn(total_timesteps=TRAINING_EPISODES, progress_bar=True)

    #
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVALUATION_EPISODES)

    result = f"{algorithm_class} on {env_name} - Mean Reward: {mean_reward} +/- {std_reward}"

    env.close()

    return result


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
