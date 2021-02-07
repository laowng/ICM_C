import os
def getfile(path, is_path=True, type="dir"):
    filelist = []
    filelist_ = os.listdir(path)
    if is_path:
        for i in range(len(filelist_)):
            filelist_[i] = os.path.join(path, filelist_[i])
    if type == "dir":
        for p in filelist_:
            if "." not in os.path.splitext(p)[1]:
                filelist.append(p)
    elif type == "pic":
        for p in filelist_:
            if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ]:
                filelist.append(p)
    elif type == "pt":
        for p in filelist_:
            if os.path.splitext(p)[1] in [".pt" ]:
                filelist.append(p)

    filelist.sort()
    return filelist