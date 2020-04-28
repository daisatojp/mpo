import argparse


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of MPO on gym environments')

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                        help='gym environment')
    parser.add_argument('--policy_evaluation', type=str, default='td',
                        help='policy evalution method')
    parser.add_argument('--dual_constraint', type=float, default=0.1,
                        help='hard constraint of the E-step')
    parser.add_argument('--kl_mean_constraint', type=float, default=0.01,
                        help='hard constraint on mean parameter')
    parser.add_argument('--kl_var_constraint', type=float, default=1e-4,
                        help='hard constraint on variance parameter')
    parser.add_argument('--kl_constraint', type=float, default=0.01,
                        help='hard constraint on variance parameter')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='scaling factor of the lagrangian multiplier in the M-step')
    parser.add_argument('--sample_process_num', type=int, default=5)
    parser.add_argument('--sample_episode_num', type=int, default=30,
                        help='number of episodes to learn')
    parser.add_argument('--sample_episode_maxlen', type=int, default=200,
                        help='length of an episode (number of training steps)')
    parser.add_argument('--sample_action_num', type=int, default=64,
                        help='number of sampled actions')
    parser.add_argument('--replay_length', type=int, default=1,
                        help='replay length')
    parser.add_argument('--replay_padding', action='store_true')
    parser.add_argument('--episode_rerun_num', type=int, default=5,
                        help='number of reruns of sampled episode')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lagrange_iteration_num', type=int, default=5,
                        help='number of optimization steps of the Lagrangian')
    parser.add_argument('--iteration_num', type=int, default=1000,
                        help='number of iteration to learn')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='log directory')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load', type=str, default=None,
                        help='load path')

    return parser.parse_args()
