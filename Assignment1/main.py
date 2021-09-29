import numpy as np

from grid_world import GridWorld
import sys

policy_threshold = 1e-3
usage = 'Simple Usage:\n' \
        'python main.py\n' \
        'Details:\n'\
        'python main.py -h'\
        'python main.py <width, 4 by default> <height, 4 by default> <accuracy, 1e-4 by default>'

if __name__ == '__main__':
    width = 4
    height = 4
    accuracy = 1e-4
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h':
            print(usage)
            exit()
        width = int(sys.argv[1])
        if len(sys.argv) > 2:
            height = int(sys.argv[2])
        if len(sys.argv) > 3:
            accuracy = float(sys.argv[3])
    grid = GridWorld(width, height, accuracy)

    policy_delta = policy_threshold + 1
    iteration = 0
    while policy_delta >= policy_threshold:
        print("====== Iteration %d ======" % iteration)
        print("Start Iterative Evaluation ...")
        grid.iterative_evaluation()
        print("Policy Evaluation Result (State Value Function):")
        print(grid.state_values)
        print("Improve Policy...")
        policy_delta = grid.improve_policy()
        print("Current Policy:")
        grid.print_policy()
        iteration += 1
        print('')
    print("Policy is stable")

