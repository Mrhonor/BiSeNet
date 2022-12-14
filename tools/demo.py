
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from time import time

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file
from tools.configer import Configer

torch.set_grad_enabled(False) 
np.random.seed(123)


# args
parse = argparse.ArgumentParser()

parse.add_argument('--weight_path', type=str, default='res/Mds/model_50000.pth',)
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_eval.json',)
parse.add_argument('--img_path', dest='img_path', type=str, default='0001TP_006720.png',)
args = parse.parse_args()
# cfg = set_cfg_from_file(args.config)
configer = Configer(configs=args.config)

# palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
labels_info_eval = [
    {"name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"name": "void", "ignoreInEval": False, "id": 34, "color": [0, 0, 0], "trainId": 19}
]

def buildPalette(labels_info):
    palette = []
    for el in labels_info:
        palette.append(el["color"])
        
    return np.array(palette)
palette = buildPalette(labels_info_eval)
# print(Palette)

class E2EModel(torch.nn.Module):
        
    def __init__(self, configer, weight_path) -> None:
        super().__init__()
        
        self.mean = torch.tensor([0.3038, 0.3383, 0.3034])[:, None, None] #.cuda()
        self.std = torch.tensor([0.2071, 0.2088, 0.2090])[:, None, None] #.cuda()
        
        # self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode="pred")
        self.net = model_factory[configer.get('model_name')](configer)
        self.net.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        self.net.eval()
        # self.net.cuda()
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.div_(255.)
        x = x.sub_(self.mean).div_(self.std).clone()
        # out = self.net(x)[0]
        out = self.net(x)
        return out
    
## mean: [0.3038, 0.3383, 0.3034] std: [0.2071, 0.2088, 0.2090]    
net = E2EModel(configer, args.weight_path)
# net.load_state_dict(torch.load('res/model_50000.pth', map_location='cpu'), strict=False)

# # define model
# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
# net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
# net.eval()
# net.cuda()

# # prepare data
# to_tensor = T.ToTensor(
#     mean=(0.3257, 0.3690, 0.3223), # city, rgb
#     std=(0.2112, 0.2148, 0.2115),
# )

print(args.img_path)
im = cv2.imread(args.img_path)[:, :, ::-1]

for i in range(1):
    t0 = time()
    # input_im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    input_im = cv2.resize(im, (960, 768))
    # input_im = im
    
    input_im = torch.tensor(input_im.astype(np.float32).copy()).unsqueeze(0) #.cuda()
    # print(input_im)
    # inference
    # out1 = net1(input_im).squeeze().detach().cpu().numpy()
    # out2 = net(input_im).long().squeeze().detach().cpu().numpy()
    out2 = net(input_im)
    # print(out2.shape)
    out2 = out2[0].long().squeeze().detach().cpu().numpy()
    
    # # print(maxV)
    # th = 0.9
    # maxV[maxV>=th] = 12
    # maxV[maxV<th] = 19
    # maxV = maxV.long().squeeze().detach().cpu().numpy()
    # # maxV = int(maxV)
    # # print(out2.shape)
    # # print(out.shape)
    # # pred1 = palette[out1]
    # # print(out2.shape)
    pred2 = palette[out2]
    # pred1 = palette[maxV]
    # print(pred2.shape)
    # print((time() - t0) * 1000)

# cv2.imwrite('./res1.jpg', pred1)
cv2.imwrite('./res.bmp', pred2)
# cv2.imwrite('./test.jpg', pred1)
