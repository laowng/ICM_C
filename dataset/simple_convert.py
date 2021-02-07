import pickle
import csv
def information_convert(file,target):
    dict={"keys":[],"data":{}}
    with open(file,encoding="utf-8-sig") as f:
        table=csv.reader(f)
        for i,linedata in enumerate(table):
            if i==0:
                for key in linedata:
                    dict["keys"].append(key)
            else:
                data=[]
                for k,data_item in enumerate(linedata):
                    data.append(data_item)
                    if k==1:
                        fileName=data_item
                assert len(data) == len(dict["keys"])
                dict["data"][fileName]=data
                print(i,dict["data"][fileName])
    with open(target,"wb") as f:
        pickle.dump(dict,f)
if __name__ == '__main__':
    information_convert("./C_data_after_process.csv","./label.pt")
    print("over")