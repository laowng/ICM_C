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
import PIL.Image as Image
from dataset import utils
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
def tensor_convert(imgPath,imgsBPath):
    transform = Compose([
        Lambda(lambda img: RResize(img)),
        FiveCrop(64),  # this is a list of PIL Images
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops],dim=1))])
    transform_Augment = Compose([
        Lambda(lambda img: RResize(img)),
        RandomVerticalFlip(0.5),
        RandomHorizontalFlip(0.5),
        RandomRotation(degrees=[-5,5]),
        Lambda(lambda img: Crop(img)),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops],dim=1))])
    with open("./label.pt","rb") as f:
        label=pickle.load(f)
    filelist = utils.getfile(imgPath, type="pic")
    num=0
    num1=0
    num2=0
    num3=0
    for i,filePath in enumerate(filelist):
        # if num==100:
        #     break
        print(i,filePath)
        dict = {}
        dict["inf"] = {}
        fileName = os.path.split(filePath)[1]
        dict["inf"]["globalID"] = label["data"][fileName][0]
        dict["inf"]["fileName"] = label["data"][fileName][1]
        dict["inf"]["Date"] = label["data"][fileName][3]
        labelName = label["data"][fileName][5]
        if labelName == "Negative ID":
            labelIndex = 0
        elif labelName == "Positive ID":
            labelIndex = 1
        elif labelName == "Unverified":
            labelIndex = -1
        elif labelName == "Unprocessed":
            shutil.copyfile(filePath, os.path.join("./unprocessedImgs", fileName))
            continue
        else:
            assert False
        dict["inf"]["labelIndex"] = labelIndex
        dict["inf"]["LatLong"] = [float(label["data"][fileName][8]), float(label["data"][fileName][9])]
        with Image.open(filePath) as img:
            imgT=transform(img)
            if imgT.size(0)!=3:
                img=img.convert("RGB")
                imgT = transform(img)
            assert imgT.size(0)==3
            dict["img"]=imgT
            with open(os.path.join(imgsBPath, "%04d.pt" % num), "wb") as imgB:
                pickle.dump(dict,imgB)
                num+=1
            if dict["inf"]["labelIndex"]==1:
                num1+=1
                for i in range(99):
                    print(i,"正样本增强")
                    imgT = transform_Augment(img)
                    if imgT.size(0) != 3:
                        img = img.convert("RGB")
                        imgT = transform(img)
                    assert imgT.size(0) == 3
                    dict["img"] = imgT
                    with open(os.path.join(imgsBPath, "%04d.pt" % num), "wb") as imgB:
                        pickle.dump(dict, imgB)
                        num += 1
                        num1 += 1
            if dict["inf"]["labelIndex"]==-1:
                num2 += 1
                for i in range(19):
                    print(i,"verifid样本增强")
                    imgT = transform_Augment(img)
                    if imgT.size(0) != 3:
                        img = img.convert("RGB")
                        imgT = transform(img)
                    assert imgT.size(0) == 3
                    dict["img"] = imgT
                    with open(os.path.join(imgsBPath, "%04d.pt" % num), "wb") as imgB:
                        pickle.dump(dict, imgB)
                        num += 1
                        num2 += 1
            else:
                num3+=1
    print("正样本:", num1, "unverified",num2,"负样本",num3)

if __name__ == '__main__':
    tensor_convert("./imgs","./imgsB")
