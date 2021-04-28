"""
Main file for TD3 using DynE embeddings.
- Use --policy_name DynE-TD3 for DynE action embeddings and raw state observations
- Use --policy_name TD3 --pixels for DynE state embeddings and raw actions ("S-DynE-TD3")
- Use --pixels --policy_name DynE-TD3 for embedded states and actions ("SA-DynE-TD3")

Adapted from Scott Fujimoto's excellent implementation: https://github.com/sfujim/TD3/
"""

import numpy as np
import torch
import gym
import argparse
import os
import time
import sys

import utils
import TD3
from EmbeddedTD3 import EmbeddedTD3
from RandomEmbeddedPolicy import RandomEmbeddedPolicy
from RandomPolicy import RandomPolicy
from embed_stack_pixel_wrapper import EmbedStackPixelObservationWrapper

# so it can find the DynE encoder and decoder
sys.path.insert(0, '../embedding')

sys.path.insert(0, '../envs')
import reacher_family

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        done = False
        while not done:
            action, *policy_state = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


# save a video of the policy
def render_policy(policy, log_dir, total_timesteps, eval_episodes=5):
    frames = []
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()

        frame = env.render(mode='rgb_array')
        frames.append(frame)
        done = False
        while not done:
            action, *policy_state = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')

            frames.append(frame)

    utils.save_gif('{}/{}.mp4'.format(log_dir, total_timesteps),
                   [torch.tensor(frame.copy()).float()/255 for frame in frames],
                   color_last=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)                         # Job name
    parser.add_argument("--policy_name", default="DynE-TD3")            # Policy name
    parser.add_argument("--env_name", default="ReacherVertical-v2")     # Environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=float)     # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e3, type=float)         # How often (time steps) we evaluate
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
    parser.add_argument("--decoder", default="default", type=str)       # Name of saved decoder
    parser.add_argument("--pixels", action="store_true")                # Use pixel observations
    parser.add_argument("--replay_size", default=1e6, type=int)         # Size of replay buffer
    parser.add_argument("--max_e_action", default=None, type=int)       # Clip the scale of the action embeddings
    parser.add_argument("--render_freq", default=5e3, type=float)       # How often (time steps) we render

    parser.add_argument("--stack", default=4, type=int)                 # frames to stack together as input
    parser.add_argument("--img_width", default=64, type=int)            # size of frames
    parser.add_argument("--source_img_width", default=64, type=int)     # size of frames before resizing

    args = parser.parse_args()
    args.save_models = not args.no_save_models

    if args.name is None:
        args.name = "{}_{}_pixel{}".format(args.env_name, args.policy_name, args.pixels)
    args.name += "_seed{}".format(args.seed)

    print("---------------------------------------")
    print("Experiment name: %s" % (args.name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.source_env is not None: source_env = args.source_env
    else: source_env = args.env_name

    if args.pixels:
        source_env = 'Pixel' + args.env_name
        # `model` contains the state encoder
        model_path = "../embedding/results/{}/{}/model_200.pt".format(args.source_env, args.decoder)
        print("Loading model from {}".format(model_path))
        model = torch.load(model_path).cuda().eval()
        state_dim = model.state_embed_size

        # renders to pixels, then encodes the pixels using `model`
        env = EmbedStackPixelObservationWrapper(env, model,
                stack=args.stack, img_width=args.img_width,
                source_img_width=args.source_img_width)
        print(model)


    # using raw actions
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
        random_policy = RandomPolicy(env.action_space)
        replay_buffer = utils.ReplayBuffer(max_size=args.replay_size)

    # using embedded actions
    elif args.policy_name == "DynE-TD3":
        # `decoder` decodes DynE actions into sequences of raw actions
        decoder_path = "../embedding/results/{}/{}/decoder.pt".format(source_env, args.decoder)
        print("Loading decoder from {}".format(decoder_path))
        decoder = torch.load(decoder_path)
        decoder.max_embedding = float(decoder.max_embedding)

        # Rescale action noise by the max embedding value since its scale was chosen
        # by TD3 based on actions in [-1, 1]
        if args.max_e_action is not None: decoder.max_embedding = min(decoder.max_embedding, args.max_e_action)
        args.policy_noise = args.policy_noise * decoder.max_embedding / decoder.traj_len
        args.expl_noise = args.expl_noise * decoder.max_embedding

        policy = EmbeddedTD3(state_dim, action_dim, max_action, decoder)

        # RandomEmbeddedPolicy takes random actions in the DynE action space
        random_policy = RandomEmbeddedPolicy(max_action, decoder)
        replay_buffer = utils.EmbeddedReplayBuffer(max_size=args.replay_size)


    print(policy.actor)
    print(policy.critic)


    # Evaluate untrained policy
    evaluations = [(0, 0, evaluate_policy(policy))]

    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_render = 0
    episode_num = 0
    done = True
    start_time = time.time()

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: {:d} Episode Num: {:d} Episode T: {:d} Reward: {:.3f} Episode time: {:.3f}s".format(
                        total_timesteps, episode_num, episode_timesteps, episode_reward, time.time() - start_time))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                             args.policy_noise, args.noise_clip, args.policy_freq)


            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append((episode_num, total_timesteps, evaluate_policy(policy)))

                if args.save_models: policy.save("policy", directory=log_dir)
                np.save("{}/eval.npy".format(log_dir), np.stack(evaluations))

            # Render current policy
            if timesteps_since_render >= args.render_freq:
                timesteps_since_render %= args.render_freq
                render_policy(policy, log_dir, total_timesteps)


            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            start_time = time.time()

            # policy is stateful because of temporally-extended actions
            policy.reset()


        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            if done: random_policy.reset()
            action, *policy_state = random_policy.select_action(np.array(obs))
        else:
            action, *policy_state = policy.select_action(np.array(obs), args.expl_noise)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env_max_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, *policy_state, reward, done_bool))

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
