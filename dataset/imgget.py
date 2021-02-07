import torch.utils.data as data
import torch
import torch.nn as nn
import os
from dataset import utils
import pickle
from torchvision.transforms import FiveCrop
from torchvision.transforms import Compose
from torchvision.transforms import Lambda
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
class Dataset(data.Dataset):
    def __init__(self, model=0):
        super(Dataset, self).__init__()
        self.model=model#0 表示 verfied分类器 1表示大黄蜂分类器
        self.data=[]
        filelist=self._scanf()
        transform = Compose([
                            FiveCrop(64),  # this is a list of PIL Images
                    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))])
        for filePath in filelist:
            data=self._getdata(filePath)
            if data["inf"]["labelIndex"]==-1 and self.model==1:
                continue
            self.data.append(data)
            if data["inf"]["labelIndex"]==1:
                pass#增强
    def __getitem__(self, index):
        target=torch.tensor([self.data[index]["inf"]["labelIndex"]])
        return self.data[index]["img"],str(self.data[index]["inf"]),target
    def _scanf(self):
        return utils.getfile("./imgsB",type="pt")
    def _getdata(self,path):
        with open(path,"rb") as f:
            data=pickle.load(f)
        return data
    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_set=Dataset()
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=10, shuffle=True)
    num1=0
    num2=0
    num3=0
    for batch in train_loader:
        input,info, target = batch[0].float(), batch[1],batch[-1].float()
        for index,imgs in enumerate(input):
            dir=None
            num=None
            if target[index][0]==-1:
                dir="./imgsunverfied"
                num=num1
                num1+=1
            elif target[index][0]==0:
                dir="./imgsNegtive"
                num=num2
                num2+=1
            elif target[index][0]==1:
                dir="./imgsPositive"
                num=num3
                num3+=1
            if num1>=500 and num2>=500 and num3>=500:
                break
            if num>=500:
                continue
            if not os.path.exists(os.path.join(dir, str(num))):
               os.mkdir(os.path.join(dir, str(num)))
            for i in range(5):
                ToPILImage()(imgs[:,i,:,:]).save(os.path.join(dir,str(num),str(i)+".jpg"))
            with open(os.path.join(dir,str(num),"info.txt"),"w") as f:
                f.writelines(info)

        #print(target)
        #print(output.shape,input.shape,LL.shape,target)