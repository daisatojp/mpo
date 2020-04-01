import argparse
import gym
from mpo_continuous import MPO


def _add_bool_arg(parser, name, default=False):
    """
    Adds a boolean argument to the parser
    :param parser: (ArgumentParser) the parser the arguments should be added to
    :param name: (str) name of the argument
    :param default: (bool) default value of the argument
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def _parse():
    """
    initialize the argument parser and parse the arguments
    :return: the dict version of the parsers output
     """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of MPO on gym environments')

    parser.add_argument(
        '--env', type=str, default='Pendulum-v0',
        help='gym environment')
    parser.add_argument(
        '--load', type=str, default=None,
        help='loading path if given')

    # train params - general
    _add_bool_arg(parser, 'train', default=True)
    _add_bool_arg(parser, 'render', default=False)
    _add_bool_arg(parser, 'save', default=True)
    _add_bool_arg(parser, 'log', default=False)
    parser.add_argument(
        '--log_dir', type=str, default=None,
        help='name of the log file')

    # train parameters
    parser.add_argument(
        '--eps', type=float, default=0.1,
        help='hard constraint of the E-step')
    parser.add_argument(
        '--eps_mean', type=float, default=0.1,
        help='hard constraint on C_mu')
    parser.add_argument(
        '--eps_sigma', type=float, default=1e-4,
        help='hard constraint on C_Sigma')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='learning rate')
    parser.add_argument(
        '--alpha', type=float, default=10,
        help='scaling factor of the lagrangian multiplier in the M-step')
    parser.add_argument(
        '--sample_episode_num', type=int, default=30,
        help='number of episodes to learn')
    parser.add_argument(
        '--sample_episode_maxlen', type=int, default=200,
        help='length of an episode (number of training steps)')
    parser.add_argument(
        '--rerun_num', type=int, default=5,
        help='number of reruns of the mini batch')
    parser.add_argument(
        '--mb_size', type=int, default=64,
        help='size of the mini batch')
    parser.add_argument(
        '--lagrange_iteration_num', type=int, default=5,
        help='number of optimization steps of the Lagrangian')
    parser.add_argument(
        '--add_act', type=int, default=64,
        help='number of additional actions')
    parser.add_argument(
        '--iteration_num', type=int, default=1000,
        help='number of iteration to learn')
    parser.add_argument(
        '--save_path', type=str, default='mpo_model.pt',
        help='saving path if save flag is set')

    # eval params (default at false)
    _add_bool_arg(parser, 'eval', default=False)
    parser.add_argument(
        '--eval_episodes', type=int, default=100,
        help='number of episodes for evaluation')
    parser.add_argument(
        '--eval_ep_length', type=int, default=3000,
        help='length of an evaluation episode')

    # parse arguments
    args = parser.parse_args()
    d_args = vars(args)
    return d_args


def main():
    model_args = _parse()
    env = gym.make(model_args['env'])
    model = MPO(env,
                dual_constraint=model_args['eps'],
                mean_constraint=model_args['eps_mean'],
                var_constraint=model_args['eps_sigma'],
                discount_factor=model_args['gamma'],
                alpha=model_args['alpha'],
                sample_episode_num=model_args['sample_episode_num'],
                sample_episode_maxlen=model_args['sample_episode_maxlen'],
                rerun_num=model_args['rerun_num'],
                mb_size=model_args['mb_size'],
                lagrange_iteration_num=model_args['lagrange_iteration_num'],
                add_act=model_args['add_act'])
    if model_args['load'] is not None:
        model.load_model(model_args['load'])
    if model_args['train']:
        model.train(
            iteration_num=model_args['iteration_num'],
            save_path=model_args['save_path'],
            log=model_args['log'],
            log_dir=model_args['log_dir'],
            render=model_args['render'])
    if model_args['eval']:
        r = model.eval(
            episodes=model_args['eval_episodes'],
            episode_length=model_args['eval_ep_length'],
            render=model_args['render'])
        r_range = env.reward_range
        print("Evaluation: mean reward = " + str(r) + ", in " +
              str(model_args['eval_episodes']) +
              " episodes(length=" + str(model_args['eval_ep_length']) +
              ", reward-range=" + str(r_range) + ")")
    env.close()


if __name__ == '__main__':
    main()
