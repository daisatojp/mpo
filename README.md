# MPO

PyTorch Implementation of the Maximum a Posteriori Policy Optimisation
([paper1](https://arxiv.org/abs/1806.06920),
[paper2](https://arxiv.org/abs/1812.02256.pdf))
Reinforcement Learning Algorithms for [OpenAI gym](https://github.com/openai/gym) environments.

## How to Run

I tested on the below environment.
* Windows 10
* Python 3.7
* PyTorch 1.8.1

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
  --log log_continuous \
  --render
```

### Discrete Action Space

```bash
python train.py \
  --device cuda:0 \
  --env LunarLander-v2 \
  --log log_discrete \
  --render
```

## License

This repository is a clone of [theogruner/rl_pro_telu](https://github.com/theogruner/rl_pro_telu),
which is licensed under the GNU GPL3 License - see the [LICENSE](LICENSE) file for details
