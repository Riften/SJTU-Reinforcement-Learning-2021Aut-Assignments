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
Iteration: 218, Delta: 0.000096
Policy Evaluation Result (State Value Function):
[[ -0. -14. -20. -22.]
 [-14. -18. -20. -20.]
 [-20. -20. -18. -14.]
 [-22. -20. -14.  -0.]]
Improve Policy...
Current Policy:
[[b'-' b'<' b'<' b'v']
 [b'^' b'^' b'v' b'v']
 [b'^' b'^' b'>' b'v']
 [b'^' b'>' b'>' b'-']]

====== Iteration 1 ======
Start Iterative Evaluation ...
Iteration: 4, Delta: 0.000000
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

Process finished with exit code 0
```
