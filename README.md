# MPO

PyTorch Implementation of the Maximum a Posteriori Policy Optimisation
([paper1](https://arxiv.org/abs/1806.06920),
[paper2](https://arxiv.org/abs/1812.02256.pdf))
Reinforcement Learning Algorithms for [OpenAI gym](https://github.com/openai/gym) environments.

## How to Run

I tested on the below environment.
* Ubuntu 18.04
* Python 3.7
* PyTorch 1.6

### INSTALL

Install PyTorch https://pytorch.org/

```bash
pip install gym Box2D IPython tqdm scipy tensorboard tensorboardx
```

### Continuous Action Space

```bash
python train.py \
  --device cuda:0 \
  --env LunarLanderContinuous-v2 \
  --dual_constraint 0.1 \
  --kl_mean_constraint 0.01 \
  --kl_var_constraint 0.0001 \
  --discount_factor 0.99 \
  --iteration_num 500 \
  --sample_process_num 5 \
  --sample_episode_num 100 \
  --sample_episode_maxlen 500 \
  --sample_action_num 64 \
  --batch_size 256 \
  --episode_rerun_num 3 \
  --log log_continuous \
  --render
```

### Discrete Action Space

```bash
python train.py \
  --device cuda:0 \
  --env LunarLander-v2 \
  --dual_constraint 0.1 \
  --kl_constraint 0.01 \
  --discount_factor 0.99 \
  --iteration_num 500 \
  --sample_process_num 5 \
  --sample_episode_num 100 \
  --sample_episode_maxlen 500 \
  --batch_size 256 \
  --episode_rerun_num 3 \
  --log log_discrete \
  --render
```

## License

This repository is a clone of [theogruner/rl_pro_telu](https://github.com/theogruner/rl_pro_telu),
which is licensed under the GNU GPL3 License - see the [LICENSE](LICENSE) file for details
