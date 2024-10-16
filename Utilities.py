import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Defines the (Torch) mse loss function
# `mse(x, y)` computes $\|x-y\|^2$
mse = nn.MSELoss()

def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    # Compute temporal difference
    target = (
        reward[1]
        + cfg.algorithm.discount_factor * target_q_values[1] * must_bootstrap[1].int()
    )
    # Compute critic loss
    critic_loss = mse(q_values[0], target)
    return critic_loss


def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """
    return -q_values[0].mean()


def get_seeds():
    return [1, 42, 123, 234, 345, 456, 567, 678, 789, 890]


def plot_rewards(library_name, all_rewards=None, directory="."):
    if all_rewards == None:
        with open(f'{directory}/{library_name}/all_rewards.pkl', 'rb') as f:
            all_rewards = pickle.load(f)
    
    stacked_rewards = torch.stack(all_rewards)
    summed_rewards = torch.sum(stacked_rewards, dim=2)
    average_rewards = torch.mean(summed_rewards, dim=0)
    std_rewards = torch.std(summed_rewards, dim=0)
    
    average_rewards_np = average_rewards.numpy()
    std_rewards_np = std_rewards.numpy()

    plt.grid(True)

    plt.plot(average_rewards_np, label='Average Summed Reward Across Seeds')
    plt.fill_between(range(len(average_rewards_np)), 
                 average_rewards_np - std_rewards_np, 
                 average_rewards_np + std_rewards_np, 
                 alpha=0.2, label='Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve: TD3 on LunarLanderContinuous-v2 (10 Seeds) - BBRL using {library_name} hyper-parameters')
    plt.legend()
    plt.show()
