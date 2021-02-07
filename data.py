import torch.utils.data as data
import torch
import torch.nn as nn
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
        if self.model==0:
            target[target >= 0] = 1
            target[target<0]=0
        return self.data[index]["img"],torch.tensor(self.data[index]["inf"]["LatLong"]),target
    def _scanf(self):
        return utils.getfile("./dataset/imgsB",type="pt")
    def _getdata(self,path):
        with open(path,"rb") as f:
            data=pickle.load(f)
        return data
    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_set=Dataset()
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=10, shuffle=False)
    for batch in train_loader:
        input,LL, target = batch[0].float(), batch[1].float(),batch[-1].float()
        conv3=nn.Conv3d(3,5,3,padding=1)
        output=conv3(input)
        #print(target)
        #print(output.shape,input.shape,LL.shape,target)