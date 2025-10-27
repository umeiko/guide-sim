from agents.ppo import Agent, ReplayData, ExperimentReplayBuffer
from networks.vit import VIT3_FC
import matplotlib.pyplot as plt
from env.guide_sim import GuidewireEnv
from env.metadata import GuideSimMetadata, HyperParams
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.transforms as transforms
import numpy as np 
import random
import os
from tqdm import tqdm
import logging
import time
import torch.multiprocessing as mp
import queue

# 日志模块初始化
log_format = "[%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
logger = logging.getLogger(__name__)
current_time = time.strftime("%m-%d_%H-%M", time.localtime())

class SubEnv:
    """
    管理每个独立进程中的游戏环境
    每个进程会维护n个环境
    通过传入的tasks列表来创建环境
    环境使用串行方式来执行
    """
    def __init__(self, tasks, hyper_params:HyperParams):
        self.tasks = tasks
        self.envs = [GuidewireEnv(task) for task in tasks]
        self.dones = [False for _ in range(len(tasks))]
        self.hyper_params = hyper_params
        for env in self.envs:
            env.hyper_params = hyper_params
        self.last_states = []
        self.episode_buffer = [ReplayData() for _ in range(len(tasks))]
        self.replay_buffer = ExperimentReplayBuffer()
        # 记录每个任务成功率，降低已经能够完成的任务的出现频率
        self.success_rate= [0 for _ in range(len(tasks))]
        self.need_reset = [False for _ in range(len(tasks))]

    # 重启环境
    def reset(self, transfrom: transforms.Compose) -> np.ndarray:
        """对每个子环境重置并且收集状态"""
        self.last_states = []
        states = None
        for k, env in enumerate(self.envs):
            s = env.reset()
            self.last_states.append(s)
            if states is None:
                states = np.zeros((len(self.tasks), *s.shape))
            states[k, ...] = s
        self.dones = [False for _ in range(len(self.tasks))]
        # 域随机化
        states = transfrom(
                torch.tensor(states, dtype=torch.uint8))
        states = states.numpy()
        return states

    # 环境执行一步
    def step(self, actions:list[int], 
             probs:list[float],
             values:list[float],
             transfrom: transforms.Compose):
        """对每个子环境串行执行操作并且收集状态"""
        states = []

        for k, env in enumerate(self.envs):
            s_next, r, d, _ = env.step(actions[k])
            self.episode_buffer[k].pack_step_data(self.last_states[k], 
                                                  actions[k], 
                                                  probs[k],
                                                  r,
                                                  values[k],
                                                  d)
            if d:  # 策略是如果结束了直接开一局新的
                s_next = env.reset()
                # 只存取第一次完成的轨迹，后面的轨迹就不存了，减少简单任务的数据出现在buffer中的比例
                if not self.need_reset[k]:
                    self.replay_buffer.pack_episode_data(self.episode_buffer[k],
                                                        self.hyper_params.lambda_,
                                                        self.hyper_params.gamma)
                self.need_reset[k] = True
            # 域随机化
            s_next = transfrom(
                    torch.tensor(s_next, dtype=torch.uint8))
            s_next = s_next.numpy()

            states.append(s_next)
        self.last_states = states
        return np.array(states)
    
    def evaler(self, agent:Agent, transfrom: transforms.Compose):
        """评估模型，对所有的子环境都进行完整交互并且返回消耗步数和奖励"""
        agent.ac_model.eval()
        states = self.reset(transfrom)
        rewards = []
        steps = []
        out_states = []
        for k, env in enumerate(self.envs):
            num_steps = 0
            for _ in range(self.hyper_params.max_steps):
                a, _,_ = agent.desision(states[k])
                state, reward, done, _ = env.step(a)

                # 域随机化
                state = transfrom(
                    torch.tensor(state, dtype=torch.uint8))
                state = state.numpy()
                # 回到numpy
                states[k] = state

                num_steps += 1
                if done:
                    logging.info(f"[eval] ID {k}: {env.task_path} done.")
                    logging.info(f"\treward: {reward}, steps: {num_steps}")
                    rewards.append(reward)
                    steps.append(num_steps)
                    out_states.append(state)
                    break     
        return out_states, np.mean(steps), np.mean(rewards)

class SubEnvMP():
    def __init__(self,
                tasks_lst:list, 
                # num_process:int,
                hyper_params:HyperParams,
                #  train_transform: transforms.Compose,
                #  eval_transform: transforms.Compose
                 ):
        num_process = hyper_params.num_processes
        assert len(tasks_lst) >= num_process, f"任务数量 ({len(tasks_lst)}) 少于进程数量 ({num_process}) "
        self.num_process = num_process
        self.tasks_per_process = []
        self.replay_buffer = ExperimentReplayBuffer()
        for i in range(num_process):
            # 将任务列表 tasks_lst 按照进程数 num_process 进行分割
            # tasks_lst[i::num_process] 表示从索引 i 开始，每隔 num_process 个元素取一个元素
            # 例如，如果 tasks_lst = [1, 2, 3, 4, 5, 6, 7, 8] 且 num_process = 3
            # 则   tasks_lst[0::3] = [1, 4, 7]
            #      tasks_lst[1::3] = [2, 5, 8]
            #      tasks_lst[2::3] = [3, 6]
            # 将分割后的任务列表添加到 self.tasks_per_process 中
            self.tasks_per_process.append(tasks_lst[i::num_process])
            logging.info(f"[SubEnvMP] process_split {i}: {tasks_lst[i::num_process]}")
        # 初始化管道, 使用zip函数将每个进程的管道对分开
        # main2sub用于主进程发送数据到子进程
        # sub2main用于子进程发送数据到主进程
        self.pipe_main2sub, self.pipe_sub2main = \
            zip(*[mp.Pipe() for _ in range(num_process)])
        
        for index in range(num_process):
            process = mp.Process(target=self.process_run, 
                                 args=(index, self.tasks_per_process[index], hyper_params))
            process.start()

        
    def step(self, acts:list[int], 
             probs:list[float],
             values:list[float],
             ):
        d_per_process = []
        a, p, v = [],[],[]
        
        for i in range(self.num_process):
            a = acts[i::self.num_process]
            p = probs[i::self.num_process]
            v = values[i::self.num_process]
            self.pipe_main2sub[i].send(("step", (a, p, v)))
            # logging.info(f"[Main] send process{i} step {a}")
        
        for i in range(self.num_process):
            d = self.pipe_main2sub[i].recv()
            d_per_process.append(d)
            # logging.info(f"[Main] get state from process{i}: {d.shape}")
        states = np.concatenate(d_per_process, axis=0)
        return states
    
    def evel(self, agent:Agent,
             
             ):
        ...
    
    def reset(self):
        logging.info(f"[Main] reset")
        states_per_process = []
        for i in range(self.num_process):
            self.pipe_main2sub[i].send(("reset", None))
        # logging.info(f"[Main] waiting reset info back")
        for i in range(self.num_process):
            states = self.pipe_main2sub[i].recv()
            # logging.info(f"[Main] get state from process{i}: {states.shape}")
            states_per_process.append(states)
        
        states = np.concatenate(states_per_process, axis=0)
        return states

    def process_run(self, index, tasks, hyper_params:HyperParams):
        # tasks里面装的是路径
        logger.info(f"[SubEnvMP] start sub process: {index}")
        se = SubEnv(tasks, hyper_params)
        while True:
            request, datas = self.pipe_sub2main[index].recv()
            if request == "step":
                batch_a, batch_p, batch_v = datas
                states = se.step(batch_a, batch_p, batch_v, transform_domain)
                self.pipe_sub2main[index].send(states)
            elif request == "reset":
                # logging.info(f"[SubEnvMP {index}] step reset")
                states = se.reset(transform_domain)
                self.pipe_sub2main[index].send(states)
            elif request == "get_replay_buffer":
                logging.info(f"[SubEnvMP {index}] get_replay_buffer {len(se.replay_buffer)}")
                self.pipe_sub2main[index].send(se.replay_buffer)
                se.replay_buffer.clear()
            # elif request == "eval":
            #     s, step, r, = se.evaler(agent, transform_norm)
    def replay_buffer_collect(self):
        for i in range(self.num_process):
            self.pipe_main2sub[i].send(("get_replay_buffer", None))
        for i in range(self.num_process):
            replay_buffer:ExperimentReplayBuffer = self.pipe_main2sub[i].recv()
            self.replay_buffer += replay_buffer

def img_ploter(states:np.ndarray) -> plt.Figure:
    """将状态画成图，用于暂存到tensorboard中"""
    num_figs = 0
    if isinstance(states, np.ndarray):
        num_figs = states.shape[0]
    elif isinstance(states, list):
        num_figs = len(states)
    num_rows = num_figs // 5 + 1
    # 生成5列n行的子图
    fig, axs = plt.subplots(num_rows, 5, figsize=(15, num_rows*3))
    now_row = 0
    for i, state in enumerate(states):
        # 每5个子图后换行
        if num_rows == 1:
            axs[i%5].imshow(state[0], cmap="gray")
            axs[i%5].axis('off')
        else:
            axs[now_row, i%5].imshow(state[0], cmap="gray")
            axs[now_row, i%5].axis('off')
        if i%5 == 4:
            now_row += 1
    return fig

# 定义转换管道，包括域随机化
transform_domain = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机改变颜色属性
    transforms.RandomResizedCrop(size= (256, 256), scale=(0.95, 1.0)),  # 随机裁剪并调整大小
    transforms.RandomRotation(degrees=2),  # 随机旋转
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]之间
]) 

# 无域随机化的普通转换管道
transform_norm = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]之间
])

def main():
    # TODO: 将该部分参数化
    # 加载任务集
    dataset_path = "datas/exvivo/"
    tasks = os.listdir(os.path.join(dataset_path, "task"))
    tasks = [os.path.join(dataset_path, "task", t) for t in tasks]
    hyper = HyperParams()
    hyper.load_from_json("./hyper.json")

    # 设置日志模块
    os.makedirs("./logs", exist_ok=True)
    logging.basicConfig(filename=f'./logs/{current_time}_{hyper.task_name}.log',
                        level=logging.INFO,format=log_format)
    writer = SummaryWriter(log_dir=f'./logs/{hyper.task_name}')

    # 初始化环境管理器（单进程）
    # TODO: 优化为多进程版本
    se = SubEnv(tasks, hyper)
    se_mp = SubEnvMP(tasks, hyper)
    # 初始化Agent
    model = VIT3_FC()
    count = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {count}")
    agent = Agent(model)
    agent.ac_model = model.to(agent.device)
    agent.set_hyperpara(hyper)
    agent.plot_attrs()
    # 加载模型权重
    weight_path = os.path.join("./weights", hyper.task_name)
    os.makedirs(weight_path, exist_ok=True)
    last_epo = 0
    if agent.load(os.path.join(weight_path, "last.pth"),'cuda:0'):
        last_epo = int(agent.epoch)
        logging.info(f'loaded weight in {os.path.join(weight_path, "last.pth")} with {last_epo} epoches')
    else:
        logging.info(f'create new weights in {os.path.join(weight_path, f"last.pth")}')
    logging.info(f'Model:\n{agent.ac_model}')

    # 初始的数据记录
    s, step, r, = se.evaler(agent, transform_norm)
    fig = img_ploter(s)
    if last_epo == 0:
        writer.add_scalar('Reward/reward', r, 0)
        writer.add_scalar('Reward/Spend Steps', step, 0)
        writer.add_figure("Last_results", fig, 
                            global_step=0)
        plt.close(fig)
    
    losses = None
    best = r
    for epoch in tqdm(range(last_epo, last_epo+hyper.num_epochs)):
        states = se_mp.reset()
        start_time = time.time()
        for _ in tqdm(range(hyper.max_steps)):
            batch_a_tensor, batch_p_tensor, batch_v_tensor, _ = agent.batch_desision(states)
            batch_a = batch_a_tensor.squeeze(1).tolist()
            batch_p = batch_p_tensor.squeeze(1).tolist()
            batch_v = batch_v_tensor.squeeze(1).tolist()
            states = se_mp.step(batch_a, batch_p, batch_v)
        
        se_mp.replay_buffer_collect()
        logging.info(f"Collected {len(se_mp.replay_buffer)} datas in buffer, cost {time.time()-start_time:.3f} s")
        if len(se_mp.replay_buffer) >= hyper.batch_size:
            losses = agent.learn(se_mp.replay_buffer)
            se_mp.replay_buffer.clear()
        
        # tensorboard
        if losses is not None:
            writer.add_scalar('Loss/actor', losses[0], epoch)
            writer.add_scalar('Loss/critic', losses[1], epoch)
            writer.add_scalar('Loss/entropy', losses[2], epoch)
            writer.add_scalar('Loss/all', losses[3], epoch)
            logging.info("[loss] [%d] actor: %.2f, critic: %.2f, entropy: %.2f, all: %.2f",
                        epoch, losses[0], losses[1], losses[2], losses[3])

        # eval
        if (epoch % hyper.plot_interval == 0) and (epoch != 0):
            s, step, r, = se.evaler(agent, transform_norm)
            fig = img_ploter(s)
            writer.add_scalar('Reward/reward', r, epoch)
            writer.add_scalar('Reward/Spend Steps', step, epoch)
            writer.add_figure("Last_results", fig,
                                global_step=epoch)
            plt.close(fig)
            if r > best:
                best = r
                agent.save(epoch, os.path.join(weight_path, f"best.pth"))
                logging.info(
                    f"save best weight in {os.path.join(weight_path, f'best.pth')} with {epoch} epoches")
            logging.info(f"[eval] of episode :{epoch}, score : {r}, steps :{step}")
        
        # checkpoints
        if ((epoch % hyper.save_interval == 0)):
            os.makedirs(weight_path, exist_ok=True)
            agent.save(epoch, os.path.join(weight_path, "last.pth"))
            logging.info("[checkpoint] last.pth saved.")


if __name__ == "__main__":
    # semp = SubEnvMP([i for i in range(27)], 3)
    # print(semp.tasks_per_process)
    main()