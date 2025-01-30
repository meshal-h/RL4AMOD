import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.nets.actor import GNNActorTD3
from src.nets.critic import GNNCriticTD3
import random
from tqdm import trange
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

import concurrent.futures
from copy import deepcopy
import threading

class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, x_s=None, reward=None, action=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    Replay buffer for SAC agents
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2):
        self.data_list.append(PairData(data1.edge_index, data1.x, torch.as_tensor(
            reward), torch.as_tensor(action), data2.edge_index, data2.x))
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False):
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=['x_s', 'x_t'])
            batch.reward = (batch.reward-mean)/(std + 1e-16)
            return batch.to(self.device)
        else:
            return Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant

#########################################
############## TD3 AGENT ################
#########################################
class TD3(nn.Module):
    def __init__(
        self,
        env,
        input_size,
        cfg, 
        parser,
        device=torch.device("cpu"),
    ):

        super(TD3, self).__init__()
        self.env = env
        self.eps = np.finfo(np.float32).eps.item(),
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion

        self.ckpt_path = cfg.ckpt_path
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.parser = parser

        # TD3 parameters
        self.max_action = 1.0
        self.min_action = 0.0 + 1e-4
        self.discount = 0.99
        self.tau = 0.1 # 0.1
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 1
        self.lr = 1.00e-3
        self.l2 = 1e-2
        # self.grad_clip = 100.0

        # Replay buffer
        self.replay_buffer = ReplayData(device=device)

        # Networks
        self.actor = GNNActorTD3(self.input_size, self.hidden_size, act_dim=self.act_dim, layer_norm=cfg.actor_layer_norm)
        self.critic_1 = GNNCriticTD3(self.input_size, self.hidden_size, act_dim=self.act_dim, layer_norm=cfg.q_layer_norm)
        self.critic_2 = GNNCriticTD3(self.input_size, self.hidden_size, act_dim=self.act_dim, layer_norm=cfg.q_layer_norm)
        
        """
        def nan_hook(self, inp, output):
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output[0]
                if len(output) > 2:
                    raise NotImplementedError("More than one output in hook.")

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    print(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
                    #raise RuntimeError("Nan detected.")
            
            return None
            
            
        for submodule in self.actor.modules():
            submodule.register_forward_hook(nan_hook)
        for submodule in self.critic_1.modules():
            submodule.register_forward_hook(nan_hook)
        for submodule in self.critic_2.modules():
            submodule.register_forward_hook(nan_hook)
        """

        self.actor_target = deepcopy(self.actor)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

        for p in self.critic_1_target.parameters():
            p.requires_grad = False
        for p in self.critic_2_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=self.l2)
        self.critic_1_optimizer = torch.optim.AdamW(self.critic_1.parameters(), lr=self.lr, weight_decay=self.l2)
        self.critic_2_optimizer = torch.optim.AdamW(self.critic_2.parameters(), lr=self.lr, weight_decay=self.l2)

        # Other
        self.directory = cfg.directory
        self.agent_name = cfg.agent_name
        self.cplexpath = cfg.cplexpath

        self.entropy_factor = cfg.entropy_factor

        self.total_it = 0

    def select_action(self, data, deterministic=True):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def update(self, data):
        self.total_it += 1

        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, self.act_dim),
        )

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action_batch) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state_batch, edge_index2, True)[0]
            next_action = (next_action + noise).clamp(self.min_action, self.max_action)
            next_action = next_action / next_action.sum(dim=-1, keepdim=True)

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state_batch, edge_index2, next_action) 
            target_Q2 = self.critic_2_target(next_state_batch, edge_index2, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state_batch, edge_index, action_batch)
        current_Q2 = self.critic_2(state_batch, edge_index, action_batch)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        critic_loss.backward()

        # with torch.autograd.detect_anomaly():
        #         critic_loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        # torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)

        # for name, param in self.critic_1.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"Critic 1.")
        #             print(f"Gradient of {name} is nan.")
        #             print(param.grad)
        
        # for name, param in self.critic_2.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"Critic 2.")
        #             print(f"Gradient of {name} is nan.")
        #             print(param.grad)

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if (self.total_it % self.policy_freq == 0):

            # Compute actor loss
            actor_action = self.actor(state_batch, edge_index, True)[0]
            q_loss = -self.critic_1(state_batch, edge_index, actor_action).mean() 
            if self.entropy_factor == 0:
                actor_loss = q_loss
            else:
                raise NotImplementedError("Entropy factor not implemented yet.")
                actor_entropy = (actor_action * actor_action.log()).sum(dim=-1)
                actor_loss = q_loss + self.entropy_factor*actor_entropy.mean()

            # actor_loss = -self.critic_1(state_batch, edge_index, self.actor(state_batch, edge_index, True)[0]).mean() 
            # - self.actor(state_batch, edge_index, True)[0].log().mean()
            # actor_loss = -self.actor(state_batch, edge_index, True)[0].log().mean()

            # actor_action = self.actor(state_batch, edge_index, True)[0]
            # actor_entropy = - actor_action * actor_action.log()
            # actor_loss = - self.critic_1(state_batch, edge_index, actor_action).mean() - actor_entropy.mean()

            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # with torch.autograd.detect_anomaly():
            #     actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any():
            #             print(f"Actor.")
            #             print(f"Gradient of {name} is nan.")
            #             print(param.grad)

            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, cfg, num):
        sim = cfg.simulator.name
        if sim == "sumo": 
            #traci.close(wait=False)
            scenario_path = '/src/envs/data/LuSTScenario/'
            sumocfg_file = 'dua_meso.static.sumocfg'
            net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')
            os.makedirs('saved_files/sumo_output/scenario_lux/', exist_ok=True)
            matching_steps = int(cfg.simulator.matching_tstep * 60 / cfg.simulator.sumo_tstep)  # sumo steps between each matching
            if 'meso' in net_file:
                matching_steps -= 1 
                
            sumo_cmd = [
            "sumo", "--no-internal-links", "-c", os.path.join(scenario_path, sumocfg_file),
            "--step-length", str(cfg.simulator.sumo_tstep),
            "--device.taxi.dispatch-algorithm", "traci",
            "-b", str(cfg.simulator.time_start * 60 * 60), "--seed", "10",
            "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(os.path.join(scenario_path, sumocfg_file)), "SUMO configuration file not found!"
        
        train_episodes = cfg.model.max_episodes  # set max number of training episodes
        epochs = trange(train_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        self.train()  # set model in train mode

        log = dict()
        log["episode"] = []
        log["reward"] = []
        log["served_demand"] = []
        log["rebalancing_cost"] = []
        log["actions"] = []
        log["avg_parameter"] = []
        log["max_parameter"] = []
        log["reb_actions"] = []
        log["acc"] = []

        nan_break = False

        # Manually create and manage the ProcessPoolExecutor
        #lock = threading.Lock()
        #executor = concurrent.futures.ProcessPoolExecutor(max_workers=cfg.other.num_update)
        
        for i_episode in epochs:

            if nan_break:
                break

            if sim =='sumo':
                traci.start(sumo_cmd)
            obs_unparsed, rew = self.env.reset()  # initialize environment
            
            log["actions"].append([])
            log["avg_parameter"].append([])
            log["max_parameter"].append([])
            log["reb_actions"].append([])

            obs = self.parser.parse_obs(obs_unparsed)
            episode_reward = 0
            episode_reward += rew
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            episode_served_demand += rew
            done = False
            if sim =='sumo' and 'meso' in net_file:
                traci.simulationStep()
            while not done:
                if sim =='sumo':
                    sumo_step = 0
                    while sumo_step < matching_steps:
                        traci.simulationStep()
                        sumo_step += 1
                
                ###########################

                ############

                if cfg.other.global_update and i_episode > 10:

                    #print("Global update")
                    
                    done = ( self.env.tf == (self.env.time+1) )
    
                    if not done:

                        if cfg.other.at_once_update:
                            buffer = ReplayData(device=self.device)

                        if cfg.other.sampling == 'dirichlet':
                            grid = generate_probability_dirichlet(len(self.env.region), cfg.other.num_update)
                        elif cfg.other.sampling == 'uniform':
                            grid = generate_probability_uniform(len(self.env.region), cfg.other.num_update)
                        else:
                            raise ValueError("Invalid sampling method.")

                        #"""
                        for n in range(cfg.other.num_update):

                            env = deepcopy(self.env)

                            obs, action_rl, rew, new_obs, nan_break = temp(env, self.parser, self.select_action, self.cplexpath, cfg.model.rew_scale, n, grid[n,:])

                            if nan_break:
                                nan_check(self.actor)
                                break

                            if cfg.other.at_once_update:
                                buffer.store(obs, action_rl, rew, new_obs)
                            else:
                                self.replay_buffer.store(obs, action_rl, rew, new_obs)
                        #"""

                        """

                        # with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.other.num_update) as executor:
                        futures = []
                        for n in range(cfg.other.num_update):
                            futures.append(
                                executor.submit(process_task, self.env, self.parser, self.select_action, self.cplexpath, cfg.model.rew_scale, self.replay_buffer, lock, n)
                            )

                        # Wait for all futures to complete
                        concurrent.futures.wait(futures)
                        
                        """

                ############
                
                if nan_break:
                    break

                obs_unparsed = (self.env.acc, self.env.time, self.env.dacc, self.env.demand)
                obs = self.parser.parse_obs(obs_unparsed)

                action_rl = self.select_action(obs)

                # check if nan 
                if np.isnan(action_rl).any():
                    print("Nan in action_rl")
                    print(action_rl)
                    print("Obs: ", obs)
                    nan_check(self.actor)
                    nan_break = True
                    break

                log["actions"][-1].append(action_rl)

                desiredAcc = {self.env.region[i]: int(action_rl[i] * dictsum(self.env.acc, self.env.time + 1))
                    for i in range(len(self.env.region))
                }

                reb_action = solveRebFlow(
                    self.env,
                    self.env.cfg.directory,
                    desiredAcc,
                    self.cplexpath,
                )
                new_obs, rew, done, info = self.env.step(reb_action=reb_action)

                log["reb_actions"][-1].append(reb_action)
                
                episode_reward += rew
                episode_served_demand += info["profit"]
                episode_rebalancing_cost += info["rebalancing_cost"]
                
                if not done: 
                    new_obs = self.parser.parse_obs(new_obs)
                    if cfg.other.global_update and i_episode > 10:
                        buffer.store(obs, action_rl, cfg.model.rew_scale * rew, new_obs)
                    else:
                        self.replay_buffer.store(obs, action_rl, cfg.model.rew_scale * rew, new_obs)

                obs = new_obs

                ###########################

                if i_episode > 10:
                    if cfg.other.at_once_update:
                        batch = buffer.sample_batch(cfg.other.num_update+1)
                        self.update(data=batch)
                    else:
                        batch = self.replay_buffer.sample_batch(cfg.model.batch_size)
                        self.update(data=batch)
                if sim =='sumo' and done:
                    traci.close()

                avg_value, max_value = parameter_value(self.actor)
                log["avg_parameter"][-1].append(avg_value)
                log["max_parameter"][-1].append(max_value)

            epochs.set_description(
                f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )

            log["episode"].append(i_episode)
            log["reward"].append(episode_reward)
            log["served_demand"].append(episode_served_demand)
            log["rebalancing_cost"].append(episode_rebalancing_cost)

            # log acc
            acc_N = len(self.env.acc)
            acc_T = len(self.env.acc[0])
            acc = np.zeros((acc_N, acc_T))

            for acc_node in range(acc_N):
                for acc_time in range(acc_T):
                    acc[acc_node, acc_time] = self.env.acc[acc_node][acc_time]

            log["acc"].append(acc)
            ###

            self.save_checkpoint(
                path=f"{self.ckpt_path}/{cfg.model.checkpoint_path}_{num}.pth"
            )
            if episode_reward > best_reward: 
                best_reward = episode_reward
                self.save_checkpoint(
                    path=f"{self.ckpt_path}/{cfg.model.checkpoint_path}_best_{num}.pth"
                )

        # Explicit shutdown after the loop completes
        #executor.shutdown(wait=True)

        if not nan_break:
            log["actions"] = np.array(log["actions"])
            log["avg_parameter"] = np.array(log["avg_parameter"])
            log["max_parameter"] = np.array(log["max_parameter"])
            log["reb_actions"] = np.array(log["reb_actions"])
            log["acc"] = np.array(log["acc"])
        else:
            print("Nan detected")
            log["actions"] = np.array(log["actions"][:-1])
            log["avg_parameter"] = np.array(log["avg_parameter"][:-1])
            log["max_parameter"] = np.array(log["max_parameter"][:-1])
            log["reb_actions"] = np.array(log["reb_actions"][:-1])
            log["acc"] = np.array(log["acc"][:-1])

        return log

    def test(self, test_episodes, env):
        sim = env.cfg.name
        if sim == "sumo":
            # traci.close(wait=False)
            os.makedirs(f'saved_files/sumo_output/{env.cfg.city}/', exist_ok=True)
            matching_steps = int(env.cfg.matching_tstep * 60 / env.cfg.sumo_tstep)  # sumo steps between each matching
            if env.scenario.is_meso:
                matching_steps -= 1

            sumo_cmd = [
                "sumo", "--no-internal-links", "-c", env.cfg.sumocfg_file,
                "--step-length", str(env.cfg.sumo_tstep),
                "--device.taxi.dispatch-algorithm", "traci",
                "--summary-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.agent_name + "_dua_meso.static.summary.xml",
                "--tripinfo-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.agent_name + "_dua_meso.static.tripinfo.xml",
                "--tripinfo-output.write-unfinished", "true",
                "-b", str(env.cfg.time_start * 60 * 60), "--seed", "10",
                "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(env.cfg.sumocfg_file), "SUMO configuration file not found!"
        epochs = trange(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_rebalanced_vehicles = []
        episode_actions = []
        episode_inflows = []
        for i_episode in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            eps_rebalancing_veh = 0
            done = False
            if sim =='sumo':
                traci.start(sumo_cmd)
            obs, rew = env.reset()  # initialize environment
            obs = self.parser.parse_obs(obs)
            eps_reward += rew
            eps_served_demand += rew
            actions = []
            inflow = np.zeros(len(env.region))
            while not done:
                
                action_rl = self.select_action(obs, deterministic=True)
                actions.append(action_rl)
                desiredAcc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(self.env.region))
                }
                reb_action = solveRebFlow(
                    self.env,
                    self.env.cfg.directory,
                    desiredAcc,
                    self.cplexpath,
                )
                new_obs, rew, done, info = env.step(reb_action=reb_action)
                #calculate inflow to each node in the graph
               
                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]

                if not done:
                    obs = self.parser.parse_obs(new_obs)
                
                eps_reward += rew
                eps_served_demand += info["profit"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                #eps_rebalancing_veh += info["rebalanced_vehicles"]
            epochs.set_description(
                f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}"
            )
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_actions.append(np.mean(actions, axis=0))
            episode_inflows.append(inflow)
            #episode_rebalanced_vehicles.append(eps_rebalancing_veh)
        

        return (
            episode_reward,
            episode_served_demand,
            episode_rebalancing_cost,
            episode_inflows,
        )

    def save_checkpoint(self, path="ckpt.pth"):
        
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
        checkpoint["critic_1_optimizer"] = self.critic_1_optimizer.state_dict()
        checkpoint["critic_2_optimizer"] = self.critic_2_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        
        checkpoint = torch.load(path)
        try:
            # Attempt to load the model state dict as is
            self.load_state_dict(checkpoint["model"])
            #print(checkpoint["model"].keys())
        except RuntimeError as e:
        
            model_state_dict = checkpoint["model"]
            new_state_dict = {}
            # Remapping the keys
            for key in model_state_dict.keys():
                if "conv1.weight" in key:
                    new_key = key.replace("conv1.weight", "conv1.lin.weight")
                #elif "lin.bias" in key:
                #    new_key = key.replace("lin.bias", "bias")
                else:
                    new_key = key
                new_state_dict[new_key] = model_state_dict[key]

            self.load_state_dict(new_state_dict)
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)


def temp(env, parser, select_action, cplexpath, rew_scale, n, acc_new):

    # Convert prob to vehicles count
    total_reb_vehicles = dictsum(env.acc, env.time + 1)

    # acc_new = (acc_new * total_reb_vehicles).round()
    # acc_diff = acc_new.sum() - total_reb_vehicles
    # i_max = np.argmax(acc_new)
    # acc_new[i_max] = acc_new[i_max] - acc_diff
    acc_new = convert_prob_to_count(acc_new, int(total_reb_vehicles))

    # Update state
    for i in env.region:
        env.acc[i][env.time + 1] = int(acc_new[i])

    # RL Section
    obs_unparsed = (env.acc, env.time, env.dacc, env.demand)
    obs = parser.parse_obs(obs_unparsed)

    action_rl = select_action(obs)

    # check if nan 
    if np.isnan(action_rl).any():
        print("Nan in action_rl")
        print(action_rl)
        print("Obs: ", obs)
        nan_break = True
        return None, None, None, None, nan_break

    desiredAcc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
        for i in range(len(env.region))
    }
    
    try:
        reb_action = solveRebFlow(
            env,
            env.cfg.directory,
            desiredAcc,
            cplexpath,
            n
        )

    except:
        print("Error in solveRebFlow; Solution does not exist.")
        print("Total Reb Vehicles: ", int(total_reb_vehicles), ", Desired Acc: ", desiredAcc)

        reb_action = [0.0]*(env.nregion*env.nregion)

    new_obs, rew, done, _ = env.step(reb_action=reb_action)
    new_obs = parser.parse_obs(new_obs)

    if done:
        raise ValueError("Environment is done, check before calling this function failed.")

    return obs, action_rl, rew_scale * rew, new_obs, False

def process_task(env, parser, select_action, cplexpath, rew_scale, replay_buffer, lock, n):
    # Copy the environment
    env_copy = deepcopy(env)

    # Perform the task
    obs, action_rl, rew, new_obs = temp(env_copy, parser, select_action, cplexpath, rew_scale, n)

    # Add transition to the replay buffer in a thread-safe manner
    with lock:
        replay_buffer.store(obs, action_rl, rew, new_obs)


def generate_probability_dirichlet(d, K):
    # Initialize the grid with the uniform probability vector
    grid = [np.ones(d) / d]
    
    # As K increases, sample from the Dirichlet distribution
    if K > 1:
        alphas = np.linspace(10, 0.1, K-1)  # Concentration parameters decreasing
        for alpha in alphas:
            grid.append(np.random.dirichlet([alpha] * d))
    
    return np.array(grid)

def generate_probability_uniform(d, k):
    # Initialize the grid
    grid=[]

    for _ in range(k):
        prob = np.random.rand(d) # Random uniform probability vector
        prob = prob / np.sum(prob)
        grid.append(prob)

    return np.array(grid)

def convert_prob_to_count(prob_vector, total_count):
    # Step 1: Scale the probabilities to the total count
    scaled_counts = prob_vector * total_count
    
    # Step 2: Floor the values to get initial integers
    int_vector = np.floor(scaled_counts).astype(int)
    
    # Step 3: Calculate the difference to be adjusted
    difference = total_count - int_vector.sum()
    
    # Step 4: Distribute the difference based on largest fractional parts
    fractional_parts = scaled_counts - int_vector
    indices = np.argsort(-fractional_parts)  # Sort in descending order of fractional part
    int_vector[indices[:difference]] += 1
    
    return int_vector

def parameter_value(model):
    total_sum = 0.0
    total_params = 0
    max_value = 0.0
    
    for param in model.parameters():
        total_sum += param.data.abs().sum().item()
        total_params += param.numel()
        max_value = max(max_value, param.data.abs().max().item())
    
    avg_value = total_sum / total_params if total_params > 0 else 0.0
    return avg_value, max_value

def nan_check(model):
    count = 0
    for param in model.parameters():
        if torch.isnan(param).any():
            count += 1
    print(f"Number of NaN parameters: {count}")
    return