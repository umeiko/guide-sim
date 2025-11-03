import json
import logging

logger = logging.getLogger(__name__)

class BaseMetadata:
    """
    Metadata class for storing GuideSimX information.
    """
    def load_from_json(self, json_path: str):
        """
        Load metadata from a json file.
        """
        # logger.info("Loading metadata from %s", json_path)
        with open(json_path, "r", encoding='utf-8') as f:
            json_dict = json.load(f)
        for key, value in json_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning("Key '%s' not found in GuideSimMetadata class.", key)

    def save_to_json(self, json_path: str):
        """
        Save metadata to a json file.
        """
        logger.info("Saving metadata to %s", json_path)
        json_dict = {}
        for key, value in self.__dict__.items():
            json_dict[key] = value
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
        logger.info("Metadata saved to %s", json_path)
    
    def __str__(self):
        """
        Return a string representation of the metadata.
        """
        out = ""
        for key, value in self.__dict__.items():
            out += f"{key}: {value}\n"
        return out


class GuideSimMetadata(BaseMetadata):
    """
    Metadata class for storing GuideSimX information.
    """
    def __init__(self):
        super().__init__()
        self.mask_path: str       = ""
        self.background_path: str = ""
        self.target_pos:list  = None
        self.direct_pos: list = None
        self.insert_pos: list = None
        self.guide_pos_lst:list = None
        self.guide_angle = None
        self.radius: float = 4.5



class HyperParams(BaseMetadata):
    """
    Hyperparameters class for storing GuideSimX information.
    """
    def __init__(self):
        super().__init__()
        # env_hyper
        self.max_steps = 80
        self.step_punishment = 0
        self.img_size = [256, 256]
        # PPO_hyper
        self.batch_size = 5
        self.beta = 0.02
        self.lr = 0.000001
        self.gamma = 0.98
        self.lambda_ = 0.99
        self.optim_type = "adam"
        self.num_epochs = 1000
        self.epsilon = 0.2
        self.device = "cuda:0"
        self.c_coef = 1.0
        self.exp_reuse_rate = 10
        self.load_opt_weight= True
        # trainer_hyper
        self.plot_interval = 10
        self.save_interval = 50
        self.num_processes = 10
        self.task_name = "task_vit"
        self.model = "VIT3_FC"
        self.task_folder_path = "./datas/train"
        self.task_num = "all"
        self.max_forward_batch = 32

