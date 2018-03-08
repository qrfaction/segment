from tool import get_files
from nipy import load_image
from setting import LABEL_PATH,IMAGE_PATH
from tqdm import tqdm
import json
def EDA():
    files = get_files()
    labels = [LABEL_PATH + f for f in files]
    images = [IMAGE_PATH + f for f in files]

    pos = {
        'x':{},
        'y':{},
        'z':{},
    }
    V = {}
    vixel = {}

    # 200 * 255 *255 *166
    for l_f,im_f in tqdm(zip(labels,images)):
        label,image = load_image(l_f),load_image(im_f)

        len_x,len_y,len_z,_ = image.shape
        max_x , min_x = 0,255
        max_y, min_y = 0, 255
        max_z, min_z = 0, 255

        for i  in range(len_x):
            for j in range(len_y):
                for k in range(len_z):
                    if label[i,j,k,0] != 0:

                        if i>max_x:
                            max_x=i
                        if i<min_x:
                            min_x=i
                        if j>max_y:
                            max_y = j
                        if j<min_y:
                            min_y=j
                        if k>max_z:
                            max_z = k
                        if k<min_z:
                            min_z=k

                        vix = image[i,j,k,0]
                        if str(i) not in pos['x']:
                            pos['x'][str(i)] = 0
                        if str(j) not in pos['y']:
                            pos['y'][str(j)] = 0
                        if str(k) not in pos['z']:
                            pos['z'][str(k)] = 0
                        if str(vix) not in vixel:
                            vixel[str(vix)] = 0
                        pos['x'][str(i)]+=1
                        pos['y'][str(j)]+=1
                        pos['z'][str(k)]+=1
                        vixel[str(vix)] +=1


        v = (max_x-min_x)*(max_y-min_y)*(max_z-min_z)
        if str(v) not in V:
            vixel[str(v)] = 0
        vixel[str(v)]+=1

    with open('pos.json','w') as f:
        f.write(json.dumps(pos,indent=4, separators=(',', ': ')))
    with open('V.json','w') as f:
        f.write(json.dumps(V,indent=4, separators=(',', ': ')))
    with open('vixel.json','w') as f:
        f.write(json.dumps(vixel,indent=4, separators=(',', ': ')))















