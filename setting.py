


SUMMARY_PATH = 'summary/'
PRE_LABEL_PATH = 'label-part1/'
PRE_IMAGE_PATH = 'atlas-part1/'
LABEL_PATH = 'label/'
IMAGE_PATH = 'image/'
MODEL_PATH = 'model/'
OUTPUT = 'output/'
INFO = 'img_info/'
BATCHSIZE = 5
WINDOW = 5
K = 10   # 交叉验证

crop_setting = {
    '3d166':{
        'x':[90,170],
        'y':[100,180],
        'z1':[40,80],
        'z2':[85,125],
    },
    '3d180':{
        'x':[100,180],
        'y':[105,185],
        'z1':[48,88],
        'z2':[90,130],
    },
    '3d160':{
        'x':[60,140],
        'y':[68,148],
        'z1':[38,78],
        'z2':[80,120],
    },

    '2d166': {
        'x': [90, 170],
        'y': [100, 180],
        'z1': [40, 80],
        'z2': [85, 125],
    },
    '2d180': {
        'x': [100, 180],
        'y': [105, 185],
        'z1': [48, 88],
        'z2': [90, 130],
    },
    '2d160': {
        'x': [60, 140],
        'y': [68, 148],
        'z1': [38, 78],
        'z2': [80, 120],
    }
}




"""
                 x1        y1       z1            x2       y2      z2         x      y         z
    180        115:161   118:172   55:81   90   115:162  112:174  99:121    37:214  36:239   28:149
    192        75:120    81:135    46:70   78   74:122   80:134   86:114    15:160  20:182   16:143
    256_166    100:163   105:173   48:75   82   98:162   107:173  90:117    25:215  28:237   19:144
    

h2  x:7->31  y: 23->46  z : 11->19
h1  x:8->31  y: 22->49  z : 11->20
h                       z : 47->65

3D :  52 * 64 * 40
2D :  64 * 100 * 100

80 * 80 * 40 

effec      46:110    42:127    23:49        46:111   43:128   63:95   
           64         85        26            65       85      32
           
           80 * 100 * 40
180        72:110    56:127            90   71:106   54:128
192        46:88     42:90     24:49   78   46:85    43:95    63:95  
256_166    100:163   105:173           82   98:162   107:173  

"""