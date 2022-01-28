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
- Install Mujoco (optional) from [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py).

## Usage
### Experiments
The common and default configurations are in `lamarckian.yml`. Subsequent experiments are inherited and modified from it. By default, all the training logs and results are stored under `~/model/lamarckian`, and can be visualized via TensorBoard.

To run training with specified algorithm and environment:
```
python train.py [-c [CONFIG ...]] [-m [MODIFY ...]] [-d] [-D] [--debug]
e.g. python train.py #with the default settings
e.g. python train.py \
	-c mdp/gym/wrap/breakout/image.yml rl/ppo.yml \# set the env and aalgorithm to use
	-m train.batch_size=4000 train.lr=0.0005 \# with modified configurtions
	-D # clear folder
```
Similarly for running evolutionary algorithms:
```
e.g. python evolve.py \
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
The `MDP` class defines the behavior of an environment. Different from the common interface as in OpenAI Gym
```python
observation, reward, done, info = env.step(action)
```
Lamarckian's asynchronous additionally defines a `Controller` .
Example of wrapping a gym environment (note that the behavior is still synchronous under the hood):
```python
from types import SimpleNamespace

class Controller:
    async def __call__(self, action):
        await self.queue.action.put(action)
        exp = await self.queue.exp.get()
        self.reward += exp.pop('reward')
        return exp
    
    def get_result(self): ... 
    def get_reward(self): ...
    def get_state(self): ...
		
class MDP:
    def __init__(self, gym_env, ...):
        self._env = gym_env
        ...
        
    def reset(self):
        states = self._env.reset()
        controllers = [Controller(self, agent_id, state) for agent_id in enumerate(states)]
        
        return SimpleNamespace(
            controlers=controlers,
            ticks=[self.tick(controllers)]
        )
    async def tick(self, controllers):
        done = False
        while not done:
            actions = [await  controller.queue.action.get() for  controller  in  controllers]
            states, rewards, done, info = self.env._step(actions)
            for  controller, state, reward  in  zip(controllers, states, rewards):
                controller.state = state
                await controller.queue.exp.put(dict(reward=reward, done=done))
            done = done  or  self.frame >= self.length
            self.frame += 1
```
#### Reinforcement Learning Algorithms
- `Actor` are the basic workers that performs sample collection and policy evaluation. An `Actor` posesses:
	- a copy of the neural network model for the policy,
	- an `Agent` class used to actually spawn agents,
	- and the `MDP` class that specifies the environment the agents will be interacting with.
- `Learner` defines the learning (policy update) process, it holds:
	- a group of (local or remote) actors to collect trajectories,
	- a neural network model instance that learns on the collected data.

Example:
```python
from contextlib import closing

class  Actor:
	...
    def  rollout(self):
        # setup the MDP environment
        loop = asyncio.get_event_loop()
        battle = self.mdp.reset(loop=loop)
        controller = battle.controllers[0]
    
        # logic of acquiring trajectory and result
        agent = self.agent_cls['train'](self.model, self.agent_config['train'])
        with closing(battle), closing(agent) as agent: # ensure the resources are released afterwards
            task = asyncio.gather(
                rollout.get_trajectory(controller, agent),
                *battle.ticks # asynchronous ticks going on in the environment
            )
            trajectory, exp = loop.run_until_complete(task)[0]
            result = controller.get_result()
        return trajectory, [result]
    
    def  __next__(self):
        trajectory, results = self.rollout()
        # accumulate the cost
        self.total_steps += sum(exp.get('step', 1) for exp in trajectory)
        return  self.to_tensor(trajectory), results


class  Learner(Actor): # the most simple Learner that directly builds on an Actor
    def _step(self):
        tensors, results = next(self)
        logp = tensors['logp'].mean(-1)
        credit = tensors['credit'].sum(-1)
        loss = (-logp * credit).mean()
        loss.backward()
        return  dict(results=results, loss=loss)
        
    def __call__(self):
        self.optimizer.zero_grad()
        outcome = self._step()
        self.optimizer.step()
        return outcome
```

#### RPC
Lamarckian builds up a Remote Procedure Call (RPC) utility with [Ray](https://docs.ray.io/en/latest/index.html) and [ZeroMQ](https://zeromq.org/languages/python/) for high-performance communication. This is useful when an RL algorithm wants to efficiently gather sampled trajectories from or broadcast the newly updated policy to some remote workers residing on different machines.

The general usage is :
```python
from lamarckian.util.rpc.wrap import all, any, gather
from lamarckian.util.rpc import All, Any, Gather
```
1. Decorate a custom class with `@all/any/gather` .
2. Call `All/Any/Gather` on a (list of) instances from this .

so that a group of sockets and background threads wiould be setup accordingly to perform remote calls.

`@all` and `All(actors:List[WrappedActor])` together specify a PUB-SUB pattern where `__call__(func_name: str, *args, **args)` invokes all remote actors' `func_name` with the passed arguments and blocks until all invocations return.
```python

@all
class  Actor:
	...
    def  set_blob(self, blob):
        state_dict = self.model.state_dict()
        state_dict = lamarckian.model.from_blob(blob, state_dict.keys(), self.device)
        self.model.load_state_dict(state_dict)

actors: List[Actor] = create_actors()
rpc_all = All(actors)
rpc_all('set_blob', blob) # to sync blob among all actors  
```

`@any` and `Any(actors:List[WrappedActor])` together specify a PUSH-PULL pattern where each remote call is dispached to any one of the available actors. The result is blockingly waited.

```python
@any
class  Actor:
	...
    def  describe(self):
        return ...
    
    def  evaluate(self, opponent):
        ... # evaluate the policy
        return results

actors: List[Actor] = create_actors()
rpc_any = Any(actors)
describe = rpc_any('describe') # get description from any of the actors
evaluate_results = rpc_any.map(('evaluate', opponent) for opponent in opponents)
```

Similarly, `@gather` and `Gather(actors:List[WrappedActor])` together specify a REQ-REP pattern.

```python
@gather
@all
class  Actor:
...
    def  gather(self):
        with torch.no_grad():
            trajectory, exp, results = self.rollout()
            cost = sum(exp.get('cost', 1) for exp in trajectory)
            tensors = self.to_tensor(trajectory, exp)
        return cost, tensors, results, self.iteration

actors: List[Actor] = create_actors()
rpc_all = All(actors)
rpc_gather = Gather(rpc_all)
rpc_gather.gathering('gather') # gather the results of actor.gather()
cost, tensors, results, iteration = rpc_gather()
```

## Reproducing the Experiments on single machine.
This section gives command of the experiments from the paper:

### Sampling Efficiency
#### PPO on Pendulum
```
python train.py \
	-c mdp/gym.yml mdp/gym/pendulum.yml rl/ppo.yml
	-m evaluator.terminate=self.cost>100000000
	-D
```
#### PPO on Gym Pong
```
python train.py \
	-c mdp/gym.yml mdp/gym/wrap/pong/image.yml rl/ppo.yml 
	-m evaluator.terminate=self.cost>100000000
	-D
```
#### PPO on Google football
```
python train.py \
	-c rl/ppo.yml mdp/gfootball.yml mdp/gfootball/simple115.yml mdp/wrap/skip.yml 
	-m evaluator.terminate=self.cost>150000000
	-D
```

### Performance and Training Speed
#### PBT+PPO on Vector Pong
```
python evolve.py \
	-c ec/ea/pbt.yml ec/wrap/mdp.yml 
	-m rl.ac.weight_loss.critic=[0,1] train.lr=[0,0.01]
	-D
```

### Evolving Behavior-diverse agents
```
python evolve.py \
	-c mdp/pong/behavior.yml ec/ea/nsga_ii.yml
	-D
```