



LABEL_PATH = 'label-part1/'
IMAGE_PATH = 'atlas-part1/'
MODEL_PATH = 'model/'
OUTPUT = 'output/'
BATCHSIZE = 5
K = 5   # 交叉验证


"""
s_x_l: 50  r: 30
s_y_l: 43  r: 40
s_z_l: 5  r: 24
x: 154   y: 182   z: 120

s_x_l: 52  r: 29
s_y_l: 47  r: 42
s_z_l: -1  r: 23
x: 147   y: 173   z: 120

s_x_l: 46  r: 30
s_y_l: 49  r: 39
s_z_l: 5  r: 26
x: 150   y: 187   z: 121

s_x_l: 49  r: 27
s_y_l: 46  r: 46
s_z_l: 1  r: 24
x: 152   y: 184   z: 123

s_x_l: 52  r: 28
s_y_l: 46  r: 46
s_z_l: 2  r: 26
x: 151   y: 203   z: 120

s_x_l: 51  r: 27
s_y_l: 42  r: 44
s_z_l: 5  r: 25
x: 155   y: 182   z: 124

s_x_l: 51  r: 29
s_y_l: 45  r: 44
s_z_l: -6  r: 25
x: 164   y: 183   z: 125

s_x_l: 48  r: 24
s_y_l: 48  r: 46
s_z_l: -4  r: 23
x: 156   y: 188   z: 121

s_x_l: 54  r: 35
s_y_l: 45  r: 44
s_z_l: -8  r: 26
x: 148   y: 186   z: 119

s_x_l: 51  r: 31
s_y_l: 47  r: 48
s_z_l: -4  r: 26
x: 160   y: 188   z: 118


s_x_l: 49  r: 29
s_y_l: 49  r: 47
s_z_l: 2  r: 26
x: 154   y: 188   z: 120


label 1    x [75:163]  y[81:173]   z[46,81]
label 2    x [74:162]  y[80:174]   z[86,121]

brain      x [15:215]  y[20:239]   z[16,149]

"""