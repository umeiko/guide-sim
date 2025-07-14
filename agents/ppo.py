import sys 
sys.path.append("..") 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
from env.metadata import HyperParams 


logger = logging.getLogger(__name__)


class ReplayData:
    """单幕交互中的数据,用来计算和记录优势值,每幕完成后需要清空"""
    def __init__(self) -> None:
        self.datas = {
                "s":[],
                "a":[],
                "p":[],
                "r":[],
                "v":[],
                "d":[],
            }
        
    def pack_step_data(self, state:np.ndarray,
                        action:int, 
                        prob:float, 
                        reward:float,
                        value:float,
                        done:bool):
        """将单次交互的结果存储
        `state` `action` `prob` `reward` `value` `done`
        """
        self.datas["s"].append(state)
        self.datas["a"].append(action)
        self.datas["p"].append(prob)
        self.datas["r"].append(reward)
        self.datas["v"].append(value)
        self.datas["d"].append(done)

    def calc_gae(self, lambda_:float, gamma:float):
        '''在单局结束时计算GAE优势,返回gae收益及gae优势'''
        r = self.datas["r"]
        d = self.datas["d"]
        v = self.datas["v"]
        next_value = 0.
        rewards = []
        advantages = []
        gae = 0
        for reward, done, value in list(zip(r, d, v))[::-1]:	
            gae = gae * lambda_ * gamma
            gae += reward + gamma * next_value * (1. - done) - value  # <-- 这个是gae优势值
            next_value = value
            advantages.insert(0, gae)       # <-- 这个是gae优势值
            rewards.insert(0, gae + value)   # <-- 这里储存的是折算后总收益，没有减去基线的
        return rewards, advantages
    
    def clear(self):
        self.datas = {
            "s":[],
            "a":[],
            "p":[],
            "r":[],
            "v":[],
            "d":[],
        }
    

class ExperimentReplayBuffer():
    """可用于训练的经验回放池"""
    def __init__(self,) -> None:
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.gaes    = []
        self.old_probs  = []
        self.advantages = []

    def pack_episode_data(self, replaydata: ReplayData, lambda_=0.95, gamma=0.99):
        """将单幕的数据打包并传入池中, 将单幕存储器清空"""
        g, adv = replaydata.calc_gae(lambda_, gamma)
        self.states += replaydata.datas["s"]
        self.actions += replaydata.datas["a"]
        self.rewards += replaydata.datas["r"]
        self.values  += replaydata.datas["v"]
        self.old_probs += replaydata.datas["p"]
        self.gaes       += g
        self.advantages += adv
        replaydata.clear()

    def get_needed_data(self):
        """返回PPO训练所需的所有数据: `states` `actions` `rewards` `probs` `values` `gaes` `advantages`"""
        out_states = np.zeros((len(self.states), *self.states[0].shape), dtype=np.float32)
        for k, state in enumerate(self.states):
            out_states[k] = state

        return out_states, np.array(self.actions).astype(np.int32), np.array(self.rewards).astype(np.float32), \
                np.array(self.old_probs).astype(np.float32), np.array(self.values).astype(np.float32),\
                np.array(self.gaes).astype(np.float32), np.array(self.advantages).astype(np.float32)

    def clear(self):
        """清空回放池"""
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.gaes    = []
        self.old_probs  = []
        self.advantages = []
    
    def __len__(self):
        return len(self.states)
    
    def __add__(self, other):
        """回放池的合并"""
        if isinstance(other, ExperimentReplayBuffer):
            self.states += other.states
            self.actions += other.actions
            self.rewards += other.rewards
            self.values  += other.values
            self.gaes    += other.gaes
            self.old_probs += other.old_probs
            self.advantages += other.advantages
        else:
            raise TypeError("unsupported operand type(s) for +: 'ExperimentReplayBuffer' and '{}'".format(type(other)))

OPT_TYPE = {
    "adam":torch.optim.Adam,
    "sgd":torch.optim.SGD,
}

class BaseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch:torch.Tensor = torch.tensor([0], dtype=torch.int32)
        # self.fc_epo = nn.Linear(1,1)
        # self.fc_epo.bias.data.fill_(0.0)

    ...

class Agent():
    def __init__(self, ac_model:BaseNetwork,
                    lr = 2e-4,          # 学习率
                    p_c = 1.0,          # Critic学习参数
                    gamma = 0.98,       # 奖励衰减系数
                    lambda_ = 1.0,      # GAE系数
                    beta = 0.01,        # 熵损失系数
                    epsilon = 0.2,      # PPO裁剪系数
                    batch_size = 16,    # 训练批量
                    exp_reuse_rate = 10,   # 经验回放复用率
                    device = None
                ) -> None:
        if device is None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.ac_model = ac_model
        self.lr = lr
        self.c_coef = p_c
        self.gamma = gamma
        self.lambda_ = lambda_
        self.beta = beta
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.exp_reuse_rate = exp_reuse_rate
        self.epoch = 0
        self.optm:torch.optim.Optimizer = None
        self.optim_type = "adam"
    
    def set_hyperpara(self, hp:HyperParams):
        self.lr = hp.lr
        self.c_coef = hp.c_coef
        self.gamma = hp.gamma
        self.lambda_ = hp.lambda_
        self.beta = hp.beta
        self.epsilon = hp.epsilon
        self.batch_size = hp.batch_size
        self.exp_reuse_rate = hp.exp_reuse_rate
        self.optim_type = hp.optim_type

    
    def plot_attrs(self):
        print("Agent's attributes:")
        print("----------------------")
        for key in dir(self):
            attr = getattr(self, key)
            if not callable(attr) and not key.startswith('__'):
                print(key, ':\t\t', attr)
        print("----------------------")

    def load(self, path, map_location):
        success = False
        try:
            # checkpoint = torch.load(path)
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
            self.epoch = checkpoint['epoch']
            self.ac_model.load_state_dict(checkpoint['model_state_dict'])
            success =  True
        
        except BaseException as e:
            logging.error(f"loading {path} with :{e}")
            success =  False
        
        try:
            if checkpoint['type_opt'] == self.optim_type:
                type_opt = checkpoint['type_opt']
                self.optm:torch.optim.Optimizer = OPT_TYPE[type_opt](self.ac_model.parameters())
                self.optm.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logging.error(f"loading Optimizer error: excepted {self.optim_type} but got checkpoint:{checkpoint['type_opt']}")
        except Exception as e:
            logging.error(f"loading Optimizer with error: {e}")
        return success 
    
    def save(self, epo:int, path):  
        if isinstance(self.optm, torch.optim.Adam):
            type_opt = "adam"
        elif isinstance(self.optm, torch.optim.SGD):
            type_opt = "sgd"
        else:
            type_opt = "sgd"
        try:
            torch.save({
                'epoch': epo,
                'type_opt': type_opt,
                'model_state_dict': self.ac_model.state_dict(),
                'optimizer_state_dict': self.optm.state_dict(),
                }, path)
            return True
        except BaseException as e:
            logging.error(f"saving {path} with :{e}")
            return False



    def desision(self, state:torch.Tensor):
        '''输入【一个】状态并获得决策，决策概率及状态价值。
        
        return:
            
            action: `int` 决策的动作结果

            predict_prob: `float`  该动作被执行的概率

            value: `float`  状态价值 (Critic 的输出)
        '''
        state:torch.Tensor = torch.Tensor(state).unsqueeze(0).to(torch.float32).to(self.device)
        predict_prob, value = self.ac_model(state)
        action = torch.multinomial(predict_prob, 1, False)[0]
        predict_prob: torch.Tensor = predict_prob.squeeze()
        value: torch.Tensor = value.squeeze()
        return int(action), float(predict_prob.squeeze()[int(action)]), float(value)

    def batch_desision(self, state:np.ndarray) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''输入【一批】状态并获得决策，决策概率及状态价值。
        
        return:
            
            action: `[batch_size, 1]`
            
            predict_prob: `[batch_size, action_num]`
            
            value: `[batch_size, 1]`
            
            probs: `[batch_size, action_num]`
        '''
        # state : [batch_size, channel, height, width]
        state:torch.Tensor = torch.Tensor(state).to(torch.float32).to(self.device)
        probs, value = self.ac_model(state)
        # predict_prob : [batch_size, action_num]
        # value : [batch_size, 1]
        action = torch.multinomial(probs, 1, False)
        predict_prob = torch.gather(probs, 1, action)
        return action, predict_prob, value, probs


    def forward_fn(self, s_batch:torch.Tensor, a_batch:torch.Tensor, 
                   gae_batch:torch.Tensor, advantages_batch:torch.Tensor, 
                   old_probs_batch:torch.Tensor):
        # print("s_batch",s_batch.shape) 2,512,512
        new_probs, values = self.ac_model(s_batch)
        # π(At|St, θ) / π_old(At|St, θ)  这个比例用来限制过大的策略更新，高效利用旧决策
        ratio = torch.gather(new_probs, 1, a_batch)
        ratio = ratio / old_probs_batch

        surr1 = ratio * advantages_batch
        # 通过裁剪 π(At|St, θ) / π_old(At|St, θ) 到1的附近，限制过大的梯度更新，来自PPO论文
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_batch
        # 更新较小的那一个梯度
        actor_loss = - torch.mean(torch.minimum(surr1, surr2))
        # 基线价值函数的损失函数，希望与GAE收益较为接近
        critic_loss = self.c_coef * F.smooth_l1_loss(gae_batch, values, reduction="mean")
        # 熵损失的计算
        # 熵损失比较大，则鼓励智能体保持足够的探索
        # 熵损失比较小，则鼓励智能体输出结果更加确定
        entropy = torch.mean(torch.sum(-new_probs * torch.log(torch.clamp(new_probs, min=1e-5)), axis=1))
        entropy_loss = -self.beta * entropy
        final_loss = actor_loss + critic_loss + entropy_loss
        return final_loss, (actor_loss, critic_loss, entropy_loss)


    def learn(self, replaybuffer:ExperimentReplayBuffer)->list[float, float, float, float]:
        """训练并更新模型参数"""
        if self.optm is None:
            self.optm = OPT_TYPE[self.optim_type](self.ac_model.parameters(), lr=self.lr)
            # self.optm = torch.optim.Adam(self.ac_model.parameters(), lr=self.lr)      # 构建优化器
            logging.info(f"building optim:{self.optm}")

        s, a, r, old_probs, value, gae, advantages,  = replaybuffer.get_needed_data()
        # print("s shape:", s.shape)
        avg_losses = [0,0,0,0]  # 将此次计算得到的平均损失输出用于记录
        times = round(self.exp_reuse_rate * len(s) / self.batch_size)
        self.ac_model.train()
        start_time = time.time()

        for _ in range(times):
            indice = torch.randperm(len(s))[:self.batch_size]  # 随机采样一部分
            # indice = list(randperm(len(s))[:self.batch_size])  # 随机采样一部分
            s_batch = torch.tensor(s[indice]).to(torch.float32).to(self.device) 
            a_batch = torch.tensor(a[indice]).unsqueeze(1).to(torch.int64).to(self.device) 
            gae_batch =  torch.tensor(gae[indice]).to(torch.float32).unsqueeze(-1).to(self.device) 
            advantages_batch =  torch.tensor(advantages[indice]).unsqueeze(-1).to(torch.float32).to(self.device)  
            old_probs_batch =  torch.tensor(old_probs[indice]).unsqueeze(-1).to(torch.float32).to(self.device)  

            loss, debug_msg = self.forward_fn(s_batch, a_batch, gae_batch, advantages_batch, old_probs_batch)
            actor_loss, critic_loss, entropy_loss = debug_msg
            self.optm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
            self.optm.step()
            # 累加损失
            avg_losses[0] += float(actor_loss) / times
            avg_losses[1] += float(critic_loss) / times
            avg_losses[2] += float(entropy_loss) / times
            avg_losses[3] += float(loss) / times
        logging.info(f"Trained with {len(replaybuffer)} datas in buffer, cost {time.time()-start_time:.3f} s")
        return avg_losses