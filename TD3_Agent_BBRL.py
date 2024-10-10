import copy

from bbrl.agents import Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer

from BBRL_Agents import ContinuousQAgent, ContinuousDeterministicActor, AddGaussianNoise

class TD3(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        # Critics
        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic-1/")
        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic-2/")

        # Target critics
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic-1/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic-2/")

        # Actor
        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        if cfg.algorithm.action_noise > 0:
            noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
            self.train_policy = Agents(self.actor, noise_agent)
        else:
            self.train_policy = self.actor

        self.eval_policy = self.actor

        # Define agents over time
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic_1 = TemporalAgent(self.critic_1)
        self.t_critic_2 = TemporalAgent(self.critic_2)
        self.t_target_critic_1 = TemporalAgent(self.target_critic_1)
        self.t_target_critic_2 = TemporalAgent(self.target_critic_2)

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_1_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_2_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_2)

        self.policy_update_counter = 0
