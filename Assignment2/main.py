from cliff_walking import *
import matplotlib.pyplot as plt


def plot_q_learning(img: plt.Axes, episodes, rewards):
    img.plot(episodes, rewards, linestyle='-', color='orange', label='QLearning')


def plot_sarsa(img: plt.Axes, episodes, rewards):
    img.plot(episodes, rewards, linestyle='-', label='Sarsa')


if __name__ == '__main__':
    cliff_env = CliffWalking()

    fig = plt.figure(num=1, figsize=(20, 5), dpi=120)
    # subfigs = fig.subfigures(nrows=2, ncols=1)
    e_0_2 = fig.add_subplot(1, 3, 1)  # epsilon = 0.5
    e_0_1 = fig.add_subplot(1, 3, 2)  # epsilon = 0.1
    e_0 = fig.add_subplot(1, 3, 3)  # epsilon = 0

    e_0_2.set_title(r'$\epsilon=0.2$')
    e_0_2.set_xlabel(r'episode')
    e_0_2.set_ylabel(r'reward')

    e_0_1.set_title(r'$\epsilon=0.1$')
    e_0_1.set_xlabel(r'episode')
    # e_0_1.set_ylabel(r'reward')

    e_0.set_title(r'$\epsilon=0$')
    e_0.set_xlabel(r'episode')
    # e_0.set_ylabel(r'reward')

    cliff_env.epsilon = 0.2
    episodes, rewards = cliff_env.q_learning(200)
    plot_q_learning(e_0_2, episodes, rewards)
    print("Policy e=0.2 QLearning")
    cliff_env.action_value.print_policy()
    episodes, rewards = cliff_env.sarsa(200)
    plot_sarsa(e_0_2, episodes, rewards)
    print("Policy e=0.2 Sarsa")
    cliff_env.action_value.print_policy()
    print()

    cliff_env.epsilon = 0.1
    episodes, rewards = cliff_env.q_learning(200)
    plot_q_learning(e_0_1, episodes, rewards)
    print("Policy e=0.1 QLearning")
    cliff_env.action_value.print_policy()

    episodes, rewards = cliff_env.sarsa(200)
    plot_sarsa(e_0_1, episodes, rewards)
    print("Policy e=0.1 Sarsa")
    cliff_env.action_value.print_policy()
    print()

    cliff_env.epsilon = 0
    episodes, rewards = cliff_env.q_learning(200)
    plot_q_learning(e_0, episodes, rewards)
    print("Policy e=0 QLearning")
    cliff_env.action_value.print_policy()

    episodes, rewards = cliff_env.sarsa(200)
    plot_sarsa(e_0, episodes, rewards)
    print("Policy e=0 Sarsa")
    cliff_env.action_value.print_policy()

    plt.legend(['QLearning', 'Sarsa'])

    plt.savefig('imgs/result.png')
