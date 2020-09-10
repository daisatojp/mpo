import multiprocessing as mp
import gym
from mpo import MPO
from argparser import parse

gym.logger.set_level(40)


def main():
    args = parse()
    env = gym.make(args.env)
    model = MPO(
        args.device,
        env,
        dual_constraint=args.dual_constraint,
        kl_mean_constraint=args.kl_mean_constraint,
        kl_var_constraint=args.kl_var_constraint,
        kl_constraint=args.kl_constraint,
        discount_factor=args.discount_factor,
        alpha=args.alpha,
        sample_process_num=args.sample_process_num,
        sample_episode_num=args.sample_episode_num,
        sample_episode_maxlen=args.sample_episode_maxlen,
        sample_action_num=args.sample_action_num,
        batch_size=args.batch_size,
        episode_rerun_num=args.episode_rerun_num,
        lagrange_iteration_num=args.lagrange_iteration_num)

    if args.load is not None:
        model.load_model(args.load)

    model.train(
        iteration_num=args.iteration_num,
        log_dir=args.log_dir,
        render=args.render,
        debug=args.debug)

    env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
