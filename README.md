# traveling-salesman-problem

巡回セールスマン問題に対する解法を実装してみる.

## 利用可能な方策
- RandomPolicy: ランダムに訪問
- GreedyPolicy: 近くの点を逐次的に選択する
- Annealing
    - TwoOptPolicy: 2-opt法
    - ThreeOptPolicy: 3-opt法
- PointerNetPolicy: 深層強化学習, https://arxiv.org/abs/1611.09940


## 環境構築
### Requirements
- docker
- docker-compose

cpu版
```
docker compose up traveling_salesman_problem-cpu
```

gpu版
```
docker compose up traveling_salesman_problem-gpu
```

## Usage
./ymlに設定ファイルを置く

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

比較実験の実行
```
python -m bin.run --config_path ./yml/sample.yml
```

### PointerNetの学習
./yml/pointernet/に設定ファイルを置く

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
学習を実行

```
python -m bin.train_pointernet
```

## 結果
- テストデータ
    - 都市数: 50
    - サンプル数: 1000

![](./resources/exp001/score.png)

![サンプル](./resources/exp001/samples.png)

# Reference
- Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940.
