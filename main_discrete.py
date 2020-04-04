import gym
from mpo_discrete import MPO
from argparser import parse


def main():
    args = parse()
    env = gym.make(args.env)
    model = MPO(
        env,
        dual_constraint=args.eps,
        kl_constraint=0.1,
        discount_factor=args.gamma,
        alpha=args.alpha,
        sample_episode_num=args.sample_episode_num,
        sample_episode_maxlen=args.sample_episode_maxlen,
        sample_action_num=args.sample_action_num,
        rerun_num=args.rerun_num,
        mb_size=args.mb_size,
        lagrange_iteration_num=args.lagrange_iteration_num)
    if args.load is not None:
        model.load_model(args.load)
    if args.train:
        model.train(
            iteration_num=args.iteration_num,
            log_dir=args.log_dir,
            render=args.render)
    if args.eval:
        r = model.eval(
            episodes=args.eval_episodes,
            episode_length=args.eval_ep_length,
            render=args.render)
        r_range = env.reward_range
        print("Evaluation: mean reward = " + str(r) + ", in " +
              str(args.eval_episodes) +
              " episodes(length=" + str(args.eval_ep_length) +
              ", reward-range=" + str(r_range) + ")")
    env.close()


if __name__ == '__main__':
    main()
