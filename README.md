# traveling-salesman-problem

Implementation of algorithms for the Traveling Salesman Problem.

## Available Policies
- RandomPolicy: Visit points randomly.
- GreedyPolicy: Visit the nearest point.
- Annealing
    - TwoOptPolicy: 2-opt algorithm
    - ThreeOptPolicy: 3-opt algorithm
- PointerNetPolicy: Deep reinforcement learning method for the TSP.

## Setup
### Requirements
- docker
- docker-compose
- nvidia-docker2 (for GPU version)

For CPU version
```
docker compose up traveling_salesman_problem-cpu
```

For GPU version
```
docker compose up traveling_salesman_problem-gpu
```

## Usage
Place the configuration file in ./yml/

The configuration file is created for Config of pytorch-pfn-extras.
See here for more details.
- [pytorch-pfn-extras Config system](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/config.html#callable-substitution)
- [みんなが知らない pytorch-pfn-extras](https://www.slideshare.net/TakujiTahara/20210618-lt-pyrotch-pfn-extras-and-config-systems-tawara)

```yml
# yml/sample.yml
exp_name: exp001

# data config
num_nodes: 50

# policies
policies:
  - type: GreedyPolicy
  - "@/annealing/two_opt"
  - "@/annealing/three_opt"
  - "@/pointernetpolicy"

# PointerNetPolicy
pointernetpolicy:
  type: PointerNetPolicy
  model:
    type: PointerNet
    input_size: "@/dim_node"
    hidden_size: 128
    num_layers: 2
    search_method: "greedy"
  ckpt_path: /workspace/results/pointernet/exp001/last.ckpt
  device: cuda:0
```

Run the comparative experiment:
```
python -m bin.run --config_path ./yml/sample.yml
```

### Training PointerNet
Place the configuration file in ./yml/pointernet/

```yml
# yml/pointernet/sample.yml
exp_name: exp001
seed: 99

# data config
num_nodes: 50

# train config
batch_size: 32
num_steps: 30000

hidden_size: 128
actor_model:
  type: PointerNet
  input_size: "@/dim_node"
  hidden_size: "@/hidden_size"
  num_layers: 2
  search_method: "probabilistic"
critic_model:
  type: TransformerCritic
  input_size: "@/dim_node"
  hidden_size: "@/hidden_size"
  num_layers: 2

# optimizer
actor_optimizer:
  type: AdamW
  params: {type: method_call, obj: "@/actor_model", method: "parameters"}
  lr: 0.0003
  weight_decay: 0.001
critic_optimizer:
  type: AdamW
  params: {type: method_call, obj: "@/critic_model", method: "parameters"}
  lr: 0.0003
  weight_decay: 0.0001
```
Run training:

```
python -m bin.train_pointernet
```

## Results
- Test Data
    - Number of Cities: 50
    - Number of Samples: 1000

![](./resources/exp001/score.png)

![](./resources/exp001/samples.png)

# Reference
- [Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940.](https://arxiv.org/abs/1611.09940)
