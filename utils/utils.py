import torch
from models.mixnet.mixnet import MixNet
from config import CFG


def get_model(model_name):
    if model_name == 'yolo5':
        #
        # checkpoint = torch.load(MODEL_PATH)
        # model.load_state_dict(checkpoint)
        pass
    elif model_name == 'mixnet_s':
        mixnet = MixNet(arch="s")
        checkpoint = torch.load(CFG.mixnet_path)
        pre_weight = checkpoint['model_state']
        model_dict = mixnet.state_dict()
        pretrained_dict = {"module." + k: v for k, v in pre_weight.items() if "module." + k in model_dict}
        model_dict.update(pretrained_dict)
        mixnet.load_state_dict(model_dict)

        return mixnet




