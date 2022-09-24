# Lamarckian: Scalable EvoRL Platform

This repository contains the code for our paper **Lamarckian: Pushing the Boundaries of Evolutionary Reinforcement
Learning towards Asynchronous Commercial Games**.

## Table of Contents
- [Lamarckian: Scalable EvoRL Platform](#lamarckian-scalable-evorl-platform)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Usage](#usage)
    - [Experiments](#experiments)
    - [Core Concepts and Customization](#core-concepts-and-customization)
      - [Asynchronous Environment Interface](#asynchronous-environment-interface)
      - [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
      - [RPC](#rpc)
  - [Reproducing the Experiments](#reproducing-the-experiments)

## Environment Setup
- Install the latest version of [PyTorch](https://pytorch.org/get-started).
- Install the dependencies via `pip install -r requirements.txt`. Currently there are some old depencies that would require legacy releases of the packages. Migration will be done in the near future.

## Usage
### Experiments
The common and default configurations are in `lamarckian.yml`. Subsequent experiments are inherited and modified from it. By default, all the training logs and results are stored under `~/model/lamarckian`, and can be visualized via TensorBoard.

To run training with specified algorithm and environment:
```
python3 train.py [-c [CONFIG ...]] [-m [MODIFY ...]] [-d] [-D] [--debug]
e.g. python3 train.py #with the default settings
e.g. python3 train.py \
	-c mdp/gym/wrap/breakout/image.yml rl/ppo.yml \# set the env and algorithm to use
	-m train.batch_size=4000 train.lr=0.0005 \# with modified configurtions
	-D # clear folder
```
Similarly for running evolutionary algorithms:
```
e.g. python3 evolve.py \
	-c 
```

For large-scale distributed training on a cluster using [Ray](https://www.ray.io/), do the following steps before running `train/evolve.py`:
1. On the head node, run
```
$ ray start --head
Local node IP: <ip_address>

...
--------------------
Ray runtime started.
--------------------

Next steps
 To connect to this Ray runtime from another node, run
 ray start --address='<ip_address>:<port>' --redis-password='5241590000000000'
...
```
2. Connect other nodes to the head node to form a Ray Cluster
```
$ ray start --address='<ip_address>:<port>' --redis-password='5241590000000000'
Local node IP: <ip_address>
--------------------
Ray runtime started.
--------------------
To terminate the Ray runtime, run
  ray stop
```
3. Cofigure the `.yml` file or use `-m` to make run-time modification
```
ray:
    address: auto
    _redis_password: '5241590000000000'
```

See [Ray Cluster Setup](https://docs.ray.io/en/latest/cluster/cloud.html#manual-cluster) for more details.

### Core Concepts and Customization
#### Asynchronous Environment Interface
The `MDP` class defines the behavior of an environment. Different from the common synchronous interface as in OpenAI Gym
```python
observation, reward, done, info = env.step(action)
```

Lamarckian's asynchronous additionally defines a `Controller`.
```python
import types

class MDP(object):
    class Controller(object):
        def __init__(self, mdp, index):
            self.mdp = mdp
            self.index = index
            
        def close(self):
            pass
        
        def get_state(self):
            vector = self.mdp.env.get_state_vector(self.index)  # shape=(C)
            image = self.mdp.env.get_state_image(self.index)  # shape=(H, W, C)
            legal = self.mdp.env.get_legal(self.index)  # bool vector
            return dict(inputs=[vector, image], legal=legal)

        async def __call__(self, action):
            done = self.mdp.env.cast(self.index, action)
            return dict(done=done)

        def get_reward(self):
            return self.mdp.env.get_reward(self.index)

        def get_result(self):
            return dict(win=self.mdp.env.get_win())

    def __init__(self, *args, **kwargs):
        self.env = Env(*args, **kwargs)

    def reset(self, *args):
        controllers = [self.Controller(self, index) for index in args]
        return types.SimpleNamespace(
            controllers=controllers,
            close=lambda: [controller.close() for controller in controllers],
        )
```

Thus controllers can be used independently in different coroutines:
```python
async def rollout(controller, agent):
    state = controller.get_state()
    rewards = []
    while True:
        action = agent(*state['inputs'])
        exp = await controller(action)
        state = controller.get_state()
        reward = controller.get_reward()
        rewards.append(reward)
        if exp['done']:
            break
    return rewards, controller.get_result()
```

## Reproducing the Experiments
This section gives command of the experiments on single machine:

### Sampling Efficiency
#### PPO on Pendulum
```
python3 train.py \
	-c mdp/gym.yml mdp/gym/pendulum.yml rl/ppo.yml \
	-m evaluator.terminate="self.cost>100000000" \
	-D
```
#### PPO on Gym Pong
```
python3 train.py \
	-c mdp/gym.yml mdp/gym/wrap/pong/image.yml rl/ppo.yml \
	-m evaluator.terminate="self.cost>100000000" \
	-D
```
#### PPO on Google football
```
python3 train.py \
	-c rl/ppo.yml mdp/gfootball.yml mdp/gfootball/simple115.yml mdp/wrap/skip.yml \
	-m evaluator.terminate="self.cost>150000000" \
	-D
```

### Performance and Training Speed
#### PBT+PPO on Vector Pong
```
python3 evolve.py \
	-c ec/ea/pbt.yml ec/wrap/mdp.yml \
	-m rl.ac.weight_loss.critic=[0,1] train.lr=[0,0.01] \
	-D
```

### Evolving Behavior-diverse agents
```
python3 evolve.py \
	-c mdp/pong/behavior.yml ec/ea/nsga_ii.yml \
	-D
```