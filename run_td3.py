import torch
import pickle
from omegaconf import OmegaConf

from bbrl_utils.nn import  soft_update_params
from bbrl.visu.plot_policies import plot_policy

from TD3_Agent_BBRL import TD3
from Utilities import compute_critic_loss, compute_actor_loss, get_seeds, plot_rewards

def run_td3(td3: TD3):
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)
        terminated, reward = rb_workspace["env/terminated", "env/reward"]
        must_bootstrap = ~terminated

        # Mise à jour des deux critiques
        td3.t_critic_1(rb_workspace, t=0, n_steps=1)
        td3.t_critic_2(rb_workspace, t=0, n_steps=1)

        with torch.no_grad():
            # Ajout de bruit pour le lissage de la politique cible
            td3.t_actor(rb_workspace, t=1, n_steps=1)
            action_t1 = rb_workspace.get("action", t=1)
            noise = torch.randn_like(action_t1) * td3.cfg.algorithm.target_noise
            noise = torch.clamp(noise, -td3.cfg.algorithm.noise_clip, td3.cfg.algorithm.noise_clip)
            noisy_action_t1 = torch.clamp(action_t1 + noise, -1, 1)
            rb_workspace.set("action", t=1, v=noisy_action_t1)

            # Calcul des Q-valeurs cibles avec les deux critiques cibles
            td3.t_target_critic_1(rb_workspace, t=1, n_steps=1)
            td3.t_target_critic_2(rb_workspace, t=1, n_steps=1)

        q_values_1, q_values_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]
        target_q_values_1, target_q_values_2 = rb_workspace["target-critic-1/q_value", "target-critic-2/q_value"]

        # Prendre la Q-valeur minimale des deux critiques cibles pour réduire l'estimation excessive
        target_q_values = torch.min(target_q_values_1, target_q_values_2)

        # Calcul et mise à jour des pertes des critiques
        critic_loss_1 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_1, target_q_values
        )
        critic_loss_2 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_2, target_q_values
        )
        td3.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        td3.critic_1_optimizer.step()

        td3.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        td3.critic_2_optimizer.step()

        # Mise à jour retardée de l'acteur
        if td3.policy_update_counter % td3.cfg.algorithm.policy_delay == 0:
            td3.t_actor(rb_workspace, t=0, n_steps=1)
            td3.t_critic_1(rb_workspace, t=0, n_steps=1)
            q_values = rb_workspace["critic-1/q_value"]
            actor_loss = compute_actor_loss(q_values)

            td3.actor_optimizer.zero_grad()
            actor_loss.backward()
            td3.actor_optimizer.step()

            # Soft update of target q function
            soft_update_params(
                td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
            )
            soft_update_params(
                td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
           )

        # Incrément du compteur de mise à jour de la politique
        td3.policy_update_counter += 1

        # Évaluation et visualisation
        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_env,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )


def start_td3_simulation(library_name = "SB3"):
    all_rewards = []

    for seed in get_seeds():
        print(f"Running TD3 using {library_name} with seed {seed}")
        cfg = OmegaConf.load(f"{library_name}_params.yaml")
        
        cfg.algorithm.seed = seed
        td3 = TD3(cfg)
        
        run_td3(td3)
        
        all_rewards.append(torch.stack(td3.eval_rewards))

        td3.visualize_best()
    
    with open(f'./{library_name}/all_rewards.pkl', 'wb') as f:
        pickle.dump(all_rewards, f)

    plot_rewards(library_name, all_rewards)
