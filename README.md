# MPO

Implementation of the Maximum A Posteriori Policy Optimization
([paper1](https://arxiv.org/abs/1806.06920),
[paper2](https://arxiv.org/abs/1812.02256.pdf))
Reinforcement Learning Algorithms for [OpenAI gym](https://github.com/openai/gym) environments.

## How to use

### Continuous Action Space

```bash
python3 main_continuous.py
  --env Pendulum-v0
  --policy_evaluation td
  --gamma 0.99
  --iteration_num 500
  --sample_episode_num 50
  --replay_length 1
  --mb_size 64
  --log_dir log_Pendulum-v0_td
  --render
```

### Discrete Action Space

```bash
python3 main_discrete.py
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

## Support

* Policy Evaluation
    - [x] 1-step TD
    - [ ] Retrace

## License

This repository is a clone of [theogruner/rl_pro_telu](https://github.com/theogruner/rl_pro_telu),
which is licensed under the GNU GPL3 License - see the [LICENSE](LICENSE) file for details
