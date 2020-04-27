# MPO

Implementation of the Maximum A Posteriori Policy Optimization
([paper1](https://arxiv.org/abs/1806.06920),
[paper2](https://arxiv.org/abs/1812.02256.pdf))
Reinforcement Learning Algorithms for [OpenAI gym](https://github.com/openai/gym) environments.

## Support

Because of multiprocessing, currently supported OS is Ubuntu only.

* Policy Evaluation
    - [x] 1-step TD
    - [ ] Retrace

## How to use

### Continuous Action Space

```bash
python3 train_continuous.py
  --device cuda:0
  --env LunarLanderContinuous-v2
  --policy_evaluation td
  --dual_constraint 0.1
  --kl_mean_constraint 0.01
  --kl_var_constraint 0.01
  --discount_factor 0.99
  --iteration_num 500
  --sample_episode_num 50
  --sample_episode_maxlen 500
  --episode_rerun_num 3
  --replay_length 1
  --batch_size 128
  --log_dir log
  --render
```

### Discrete Action Space

```bash
python3 train_discrete.py
  --env LunarLander-v2
  --policy_evaluation td
  --gamma 0.99
  --iteration_num 500
  --sample_episode_num 100
  --replay_length 1
  --mb_size 128
  --log_dir log_lunarlander-v2_td
  --render
```

## License

This repository is a clone of [theogruner/rl_pro_telu](https://github.com/theogruner/rl_pro_telu),
which is licensed under the GNU GPL3 License - see the [LICENSE](LICENSE) file for details
