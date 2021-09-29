# Reinforcement Learning Assignment 1 - Small Gridworld Problem
**Code has been uploaded to [this github repo](https://github.com/Riften/SJTU-Reinforcement-Learning-2021Aut-Assignments/tree/master/Assignment1).**
## Environment
- python interpreter: Python3.8
- dependency: numpy

## Usage
Simple usage:
```bash
python main.py
```

If you want to solve the problem with different grid size or evaluation accuracy:
```bash
python main.py <width, 4 by default> <height, 4 by default> <accuracy, 1e-4 by default>
# for example
python main.py 6 5 0.01
```

## Implementation
The core implementation is in file `grid_world.py`. It defines class `GridWorld` as
the problem-solving environment. The attributes and methods of `GridWorld` are
- `state_values`: The value of state value function for all states. It is a 2D numpy array
where each element contains the state value for one state.
- `policy`, `policy_north`, `policy_east`, `policy_south`, `policy_west`: Policy. The
reason I define multiple variables here is for computation convenience.
- `evaluation()`: Compute `state_values` as policy evaluation. It will return the difference
between computed `state_values` and its original value before computation, which is 
used as the flag for evaluation iteration.
- `iterative_evaluation()`: Compute `state_values` iteratively until it is convergent.
- `improve_policy()`: Improve the current policy according to `state_values`.
- `print_policy()`: Print out the current policy in symbols `{^, >, v, <}` which represents
`{N, E, S, W}`.

## Output
### Default output
```bash
python main.py
====== Iteration 0 ======
Start Iterative Evaluation ...
Iteration: 168, Delta: 0.000098
Policy Evaluation Result (State Value Function):
[[ -0.         -10.9999284  -15.49989625 -16.49988834]
 [-10.9999284  -14.49990416 -15.99989259 -15.49989625]
 [-15.49989625 -15.99989259 -14.49990416 -10.9999284 ]
 [-16.49988834 -15.49989625 -10.9999284   -0.        ]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'v']
 [b'^' b'<' b'v' b'v']
 [b'^' b'^' b'>' b'v']
 [b'>' b'>' b'>' b'-']]

====== Iteration 1 ======
Start Iterative Evaluation ...
Iteration: 4, Delta: 0.0000004
Policy Evaluation Result (State Value Function):
[[-0. -1. -2. -3.]
 [-1. -2. -3. -2.]
 [-2. -3. -2. -1.]
 [-3. -2. -1. -0.]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'v']
 [b'^' b'^' b'^' b'v']
 [b'^' b'^' b'>' b'v']
 [b'^' b'>' b'>' b'-']]

====== Iteration 2 ======
Start Iterative Evaluation ...
Iteration: 1, Delta: 0.000000
Policy Evaluation Result (State Value Function):
[[-0. -1. -2. -3.]
 [-1. -2. -3. -2.]
 [-2. -3. -2. -1.]
 [-3. -2. -1. -0.]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'v']
 [b'^' b'^' b'^' b'v']
 [b'^' b'^' b'>' b'v']
 [b'^' b'>' b'>' b'-']]

Policy is stable

```

### Solving Small Gridworld Problem with Size 6 x 5
```bash
python main.py 6 5
====== Iteration 0 ======
Start Iterative Evaluation ...
Iteration: 476, Delta: 0.000098
Policy Evaluation Result (State Value Function):
[[ -0.         -23.52656572 -35.52338734 -41.34468167 -43.27257374]
 [-23.47327444 -32.05631606 -38.69892429 -42.23809532 -43.20047382]
 [-35.36351349 -38.52651129 -40.97791255 -41.70831705 -41.09076435]
 [-41.09076435 -41.70831705 -40.97791255 -38.52651129 -35.36351349]
 [-43.20047382 -42.23809532 -38.69892429 -32.05631606 -23.47327444]
 [-43.27257374 -41.34468167 -35.52338734 -23.52656572  -0.        ]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'<' b'<']
 [b'^' b'<' b'<' b'<' b'v']
 [b'^' b'^' b'<' b'v' b'v']
 [b'^' b'^' b'>' b'v' b'v']
 [b'^' b'>' b'>' b'>' b'v']
 [b'>' b'>' b'>' b'>' b'-']]

====== Iteration 1 ======
Start Iterative Evaluation ...
Iteration: 5, Delta: 0.00000027
Policy Evaluation Result (State Value Function):
[[-0. -1. -2. -3. -4.]
 [-1. -2. -3. -4. -4.]
 [-2. -3. -4. -4. -3.]
 [-3. -4. -4. -3. -2.]
 [-4. -4. -3. -2. -1.]
 [-4. -3. -2. -1. -0.]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'<' b'<']
 [b'^' b'^' b'^' b'^' b'v']
 [b'^' b'^' b'^' b'>' b'v']
 [b'^' b'^' b'>' b'>' b'v']
 [b'^' b'>' b'>' b'>' b'v']
 [b'>' b'>' b'>' b'>' b'-']]

====== Iteration 2 ======
Start Iterative Evaluation ...
Iteration: 1, Delta: 0.000000
Policy Evaluation Result (State Value Function):
[[-0. -1. -2. -3. -4.]
 [-1. -2. -3. -4. -4.]
 [-2. -3. -4. -4. -3.]
 [-3. -4. -4. -3. -2.]
 [-4. -4. -3. -2. -1.]
 [-4. -3. -2. -1. -0.]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'<' b'<']
 [b'^' b'^' b'^' b'^' b'v']
 [b'^' b'^' b'^' b'>' b'v']
 [b'^' b'^' b'>' b'>' b'v']
 [b'^' b'>' b'>' b'>' b'v']
 [b'>' b'>' b'>' b'>' b'-']]

Policy is stable

```