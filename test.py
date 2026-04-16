import gymnasium as gym
import metaworld

env = gym.make("Meta-World/MT1", env_name="reach-v3")

obs, info = env.reset()
print("obs shape:", obs.shape)
print("info:", info)

action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

print("next_obs shape:", next_obs.shape)
print("reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)
print("info:", info)

env.close()