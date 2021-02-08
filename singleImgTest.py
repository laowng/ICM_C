import pickle
import os
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.transforms import Resize
from torch.nn.functional import interpolate
from torchvision.transforms import FiveCrop
from torchvision.transforms import Compose
from torchvision.transforms import Lambda
from torchvision.transforms import RandomRotation
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import RandomCrop
import torch
import shutil
import numpy as np
import random
import PIL.Image as Image
from dataset import utils
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def Crop(img):
    rc=RandomCrop(64)
    crops=[]
    for i in range(5):
        crops.append(rc(img))
    return crops
def RResize(img):
    if img.size[1]>img.size[0]:
        size=(100,int(img.size[1]/(img.size[0]/100)))
    else:
        size = (int(img.size[0] / (img.size[1] / 100)),100)
    return img.resize(size,resample=Image.BICUBIC)
def Crop(img):
    rc=RandomCrop(64)
    crops=[]
    for i in range(5):
        crops.append(rc(img))
    return crops
def tensor_convert(imgPath):
    transform = Compose([
        Lambda(lambda img: RResize(img)),
        FiveCrop(64),  # this is a list of PIL Images
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops],dim=1))])
    transform_Augment = Compose([
        Lambda(lambda img: RResize(img)),
        RandomVerticalFlip(0.5),
        RandomHorizontalFlip(0.5),
        RandomRotation(degrees=[-5, 5]),
        Lambda(lambda img: Crop(img)),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops], dim=1))])
    with Image.open(imgPath) as img:
        imgT=transform(img)
        if imgT.size(0)!=3:
            img=img.convert("RGB")
            imgT = transform(img)
        assert imgT.size(0)==3
        return imgT

if __name__ == '__main__':
    setup_seed(0)
    names=["ATT2_HoireRDBeekeeperReport.jpg","ATT185_3FF5F030-F6FA-4F4E-AFE9-E90A979C9C7B.jpg","ATT19_IMG_2858.jpg","ATT1_DSCN9647.jpg","ATT12_35736840-588C-47B3-B81B-454ABFAE6A53.jpg","ATT15_B83CA488-6286-4934-BA68-A0BD5DF03D1F.jpg",
           "ATT758_30551408-D0EE-4C0F-8D8A-4EF77DB4CC18.jpg","ATT766_Harvey3.jpg","ATT769_IMG_2841.jpg","ATT770_IMG_2840.jpg","ATT773_IMG_3108.jpg","ATT778_20200708_172852.jpg",
           "ATT780_P1010569.jpg","ATT785_84EA1779-4B9C-4BFB-BE41-A579AE9E59CD.jpg"]
    model0 = torch.load("./checkpoint/model_best_0.pth")
    model1 = torch.load("./checkpoint/model_best.pth")

    input = torch.stack([tensor_convert(os.path.join("./dataset/imgs",name)) for name in names]).cuda()
    predict0 = model0(input)
    predict1=model1(input)
    for i,p in enumerate(predict0):
       print(names[i],"isVerified:",predict0[i][0].item(),"isBigBee:",predict1[i][0].item())
