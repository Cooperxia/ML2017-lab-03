
import feature
import numpy as np
import pickle
from PIL import Image
import os

def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    # write your code here
    path1="D:/datasets/original/face"#--------------This is the picture(face) location
    path2="D:/datasets/original/nonface"#-----------This is the picture(nonface) location
    files=os.listdir(path1)
    os.chdir(path1)
    for file in files:
        img=Image.open(file).convert('L')
        img=img.resize((24,24))
        npd=NPDFeature(np.array(img))
        file=file.strip(".jpg")
        save(npd.extract(),"D:/datasets/NPD_face/data_"+file)#---------------This is data(face)'s location
    
    files=os.listdir(path2)
    os.chdir(path2)
    for file in files:
        img=Image.open(file).convert('L')
        img=img.resize((24,24))
        npd=NPDFeature(np.array(img))
        file=file.strip(".jpg")
        save(npd.extract(),"D:/datasets/NPD_nonface/data_"+file)#------------This is data(nonface)'s location
        
    pass