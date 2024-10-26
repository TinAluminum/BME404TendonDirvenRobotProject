import gymnasium as gym
from stable_baselines3 import A2C

# LOG 23Oct24: got the program to run but error at the end after some training line 16
#               ValueError: You have passed a tuple to the predict() function instead of a Numpy array or a Dict. You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()` (SB3 VecEnv). See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api


# Initialise the environment
env = gym.make("LunarLander-v2", render_mode="human")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)


# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy

    observation, reward, terminated, truncated, info = env.step(model.predict(env.reset())[0])

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()