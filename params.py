import json
import os

class BaseParams:
    name = "base"
    def __init__(self) -> None:
        ...
    
    def load_from_path(self, path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError(f"\"{path}\" is not a directory")
        else:
            raise ValueError(f"\"{path}\" does not exist")
        
        f_name = os.path.join(path, f"{self.name}.json")
        if not os.path.exists(f_name):
            self.save_to_json(path)
            print(f"generated config file: \"{f_name}\"")
        else:
            with open(f_name, 'r') as file:
                data:dict = json.load(file)
            
            for key, value in data.items():
                # if hasattr(self, key):
                setattr(self, key, value)
    
    def load_from_json(self, path):
        if not os.path.exists(path):
            raise ValueError(f"\"{path}\" does not exist")
        else:
            with open(path, 'r') as file:
                data:dict = json.load(file)        
            for key, value in data.items():
                # if hasattr(self, key):
                setattr(self, key, value)

    def save_to_json(self, path):
        os.makedirs(path, exist_ok=True)
        f_name = os.path.join(path, f"{self.name}.json")
        data = {key: getattr(self, key) for key in dir(self) if not callable(getattr(self, key)) and not key.startswith('__')}
        with open(f_name, 'w') as file:
            json.dump(data, file, indent=4)
    
    def __add__(self, other):
        if isinstance(other, BaseParams):
            for key in dir(other):
                if not callable(getattr(other, key)) and not key.startswith('__'):
                    setattr(self, key, getattr(other, key))
    
    def get_msg(self) -> str:
        out = ""
        for key in dir(self):
            if not key.startswith('__'):
                if not callable(getattr(self, key)):
                    out += f'\t"{key}"={getattr(self, key)}\n'
        return out
    
    def apply(self, obj):
        """将自身参数应用与某个对象上"""
        for key in dir(obj):
            if key in dir(self) and not callable(getattr(self, key)) and not key.startswith('__'):
                setattr(obj, key, getattr(self, key))
    
    def print(self):
        """打印参数"""
        for key in dir(self):
            if not key.startswith('__'):
                if not callable(getattr(self, key)):
                    print(f'"{key}"={getattr(self, key)}')

def set_param(agent, param):
    # 读取参数
    for key in dir(param):
        if not callable(getattr(param, key)) and not key.startswith('__'):
            setattr(agent, key, getattr(param, key))


class TrainParams(BaseParams):
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    lr     = 1e-4  # learning rate
    c_coef = 1.0   # critc loss coefficient
    beta   = 1e-3  # entropy coefficient
    gamma  = 0.99
    lambda_ = 0.95
    epsilon = 0.2
    batch_size = 32
    num_epochs = 1000
    exp_reuse_rate = 5
    weights_path = None
    name = "train"



class EnvParams(BaseParams):
    image_size = (600, 700)
    actions  = None
    max_steps  = None
    input_dense = None
    name = "enviroment"
    


class RuntimeParams(BaseParams):
    num_processes = 2
    save_interval = 50
    plot_interval = 15
    task_name     = "task1"
    name = "runtime"


if __name__ == "__main__":
    test = TrainParams()
    test.save_to_json("./test.json")