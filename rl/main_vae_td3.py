"""
Main file for TD3 using a VAE to encode pixel observations.
Adapted from Scott Fujimoto's excellent implementation: https://github.com/sfujim/TD3/
"""

import numpy as np
import torch
import gym
import argparse
import os
import sys

import utils
import TD3
from embed_pixel_wrapper import EmbedPixelObservationWrapper

# so it can find the vae class
sys.path.insert(0, '../embedding')

sys.path.insert(0, '../envs')
import reacher_family

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def render_policy(policy, log_dir, total_timesteps, eval_episodes=5):
    frames = []
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        frames.append(env.render(mode='rgb_array'))
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')
            frames.append(frame)

    utils.save_gif('{}/{}.mp4'.format(log_dir, total_timesteps),
                   [torch.tensor(frame.copy()).float()/255 for frame in frames],
                   color_last=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="VAE-TD3")             # Policy name
    parser.add_argument("--env_name", default="ReacherVertical-v2")     # Environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=float)   # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=float)     # Max time steps to run environment for
    parser.add_argument("--no_save_models", action="store_true")        # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)      # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates

    parser.add_argument("--source_env", default=None)                   # env name to take the decoder from
    parser.add_argument("--decoder", default=None, type=str)            # Name of saved decoder
    parser.add_argument("--replay_size", default=1e6, type=int)         # Size of replay buffer
    parser.add_argument("--render_freq", default=5e3, type=float)       # How often (time steps) we render
    parser.add_argument("--stack", default=4, type=int)                 # frames to stack together as input
    parser.add_argument("--source_img_width", default=64, type=int)     # size of frames before resizing

    args = parser.parse_args()
    args.save_models = not args.no_save_models

    if args.name is None:
        args.name = "{}_{}".format(args.env_name, args.policy_name)
    args.name += "_seed{}".format(args.seed)

    print("---------------------------------------")
    print("Experiment name: %s" % (args.name))
    print("---------------------------------------")


    env = gym.make(args.env_name)
    env_max_steps = env._max_episode_steps

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # make a results directory and log the arguments
    log_dir = "results/{}/".format(args.name)
    os.makedirs(log_dir, exist_ok=True)
    utils.write_options(args, log_dir)

    # `model` is a VAE to use to encode the images
    model_path = "../action-embedding/results/{}/{}/model_100.pt".format(args.source_env, args.decoder)
    print("Loading model from {}".format(model_path))
    model = torch.load(model_path).cuda().eval()

    # renders to pixels, then encodes the pixels using `model`
    env = EmbedPixelObservationWrapper(env, model, stack=args.stack, source_img_width=args.source_img_width)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = TD3.TD3(state_dim, action_dim, max_action)
    replay_buffer = utils.ReplayBuffer(max_size=args.replay_size)

    print(model)
    print(policy.actor)
    print(policy.critic)


    # Evaluate untrained policy
    evaluations = [(0, 0, evaluate_policy(policy))]

    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_render = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append((episode_num, total_timesteps, evaluate_policy(policy)))

                if args.save_models: policy.save("policy", directory=log_dir)
                np.save("{}/eval.npy".format(log_dir), np.stack(evaluations))

            if timesteps_since_render >= args.render_freq:
                timesteps_since_render %= args.render_freq
                render_policy(policy, log_dir, total_timesteps)

            # Reset environment
            obs = env.reset()
            policy.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env_max_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_render += 1

    # Final evaluation
    evaluations.append((episode_num, total_timesteps, evaluate_policy(policy)))
    np.save("{}/eval.npy".format(log_dir), np.stack(evaluations))
    render_policy(policy, log_dir, total_timesteps)
    if args.save_models: policy.save("policy", directory=log_dir)
