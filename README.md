# traveling-salesman-problem

Implementation of algorithms for the Traveling Salesman Problem.

## Available Policies
- RandomPolicy: Visit points randomly.
- GreedyPolicy: Visit the nearest point.
- Annealing
    - TwoOptPolicy: 2-opt algorithm
    - ThreeOptPolicy: 3-opt algorithm
- [PointerNetPolicy]((https://arxiv.org/abs/1611.09940)): Deep reinforcement learning method for the TSP.
  - PointerNet: Vanilla PointerNet
  - TransformerPointerNet: Transformer-based PointerNet

## Setup
### Requirements
- docker
- docker-compose
- nvidia-docker2 (for GPU version)

For CPU version
```
docker compose up cpu
```

For GPU version
```
docker compose up gpu
```

## Usage
Place the configuration file in ./yml/

The configuration file is created for Config of pytorch-pfn-extras.
See here for more details.
- [pytorch-pfn-extras Config system](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/config.html#callable-substitution)
- [みんなが知らない pytorch-pfn-extras](https://www.slideshare.net/TakujiTahara/20210618-lt-pyrotch-pfn-extras-and-config-systems-tawara)

```yaml
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
python -m bin.run -c ./yml/sample.yml
```

### Training PointerNet
Place the configuration file in ./yml/pointernet/

```yaml
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
python -m bin.train_pointernet -c ./yml/pointernet/sample.yml
```

### Hyperparameter Search for Annealing
Place the configuration file in ./yml/annealing/

```yaml
# yml/tune/sample.yml
seed: 342
num_trials: 100
num_samples: 500

# generator
generator:
  type: Generator
  num_nodes: 50
  dim_node: 2

# policy
timelimit: 0.5
start_tmp: 0.01
end_tmp: 0.001
policy:
  type: TwoOptPolicy
  start_tmp: "@/start_tmp"
  end_tmp: "@/end_tmp"
  timelimit: "@/timelimit"
```

Run hyperparameter search:
```
python -m bin.tune_sa -c ./yml/tune/sample.yml
```

## Results
- Test Data
    - Number of Cities: 50
    - Number of Samples: 1000

![](./resources/exp001/score.png)

![](./resources/exp001/samples.png)

# Reference
- [Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940.](https://arxiv.org/abs/1611.09940)
