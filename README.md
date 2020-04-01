# MPO

Implementation of the Maximum A Posteriori Policy Optimization
([paper1](https://arxiv.org/abs/1806.06920), [paper2](https://arxiv.org/abs/1812.02256.pdf))
Reinforcement Learning Algorithms for continuous  control
on [OpenAI gym](https://github.com/openai/gym) environments.

## How to use

```bash
python3 main_continuous.py --env Pendulum-v0 --iteration_num 300 --log --log_dir log --render
```

## Support

* Policy Evaluation
    - [x] 1-step TD
    - [ ] Retrace

## License

This repository is a clone of [theogruner/rl_pro_telu](https://github.com/theogruner/rl_pro_telu),
which is licensed under the GNU GPL3 License - see the [LICENSE](LICENSE) file for details
