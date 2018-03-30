from tool import get_files
from nipy import load_image
from setting import LABEL_PATH,IMAGE_PATH,PRE_IMAGE_PATH,PRE_LABEL_PATH,INFO
from tqdm import tqdm
import json
import multiprocessing as mlp
import numpy as np
import SimpleITK as sitk

def image_anaylizer(images,labels):
    pos = {
        'x1': {},
        'y1': {},
        'z1': {},
        'x2': {},
        'y2': {},
        'z2': {},
        'brain_x': {},
        'brain_y': {},
        'brain_z': {},
    }
    vixel = {
        'vixel1': {},
        'vixel2': {},
        'brain_vixel': {},
    }
    effec_range = {}

    max_size_x = 0
    max_size_y = 0
    max_size_z = 0

    min_shift_x_l = 256
    min_shift_x_r = 256
    min_shift_y_l = 256
    min_shift_y_r = 256
    min_shift_z_l = 256
    min_shift_z_r = 256
    # 200 * 255 *255 *166
    for l_f, im_f in tqdm(zip(labels, images)):
        label, image = load_image(l_f).get_data(), load_image(im_f).get_data()

        len_x, len_y, len_z, _ = image.shape

        cood_x_l = 256
        cood_y_l = 256
        cood_z_l = 256
        cood_x_r = 0
        cood_y_r = 0
        cood_z_r = 0

        hippo_x_l = 256
        hippo_y_l = 256
        hippo_z_l = 256
        hippo_x_r = 0
        hippo_y_r = 0
        hippo_z_r = 0

        for i in range(len_x):
            for j in range(len_y):
                for k in range(len_z):
                    vix = image[i, j, k, 0]

                    if label[i, j, k, 0] != 0:
                        dot_x = 'x' + str(label[i, j, k, 0])
                        dot_y = 'y' + str(label[i, j, k, 0])
                        dot_z = 'z' + str(label[i, j, k, 0])
                        vix_id = 'vixel' + str(label[i, j, k, 0])
                        if int(i) not in pos[dot_x]:
                            pos[dot_x][int(i)] = 0
                        if int(j) not in pos[dot_y]:
                            pos[dot_y][int(j)] = 0
                        if int(k) not in pos[dot_z]:
                            pos[dot_z][int(k)] = 0
                        if int(vix) not in vixel[vix_id]:
                            vixel[vix_id][int(vix)] = 0
                        pos[dot_x][int(i)] += 1
                        pos[dot_y][int(j)] += 1
                        pos[dot_z][int(k)] += 1
                        vixel[vix_id][int(vix)] += 1

                        if i<hippo_x_l:
                            hippo_x_l = i
                        if i>hippo_x_r:
                            hippo_x_r = i
                        if j<hippo_y_l:
                            hippo_y_l = j
                        if j>hippo_y_r:
                            hippo_y_r = j
                        if k<hippo_z_l:
                            hippo_z_l = k
                        if k>hippo_z_r:
                            hippo_z_r = k


                    elif image[i, j, k, 0] != -1:
                        if int(i) not in pos['brain_x']:
                            pos['brain_x'][int(i)] = 0
                        if int(j) not in pos['brain_y']:
                            pos['brain_y'][int(j)] = 0
                        if int(k) not in pos['brain_z']:
                            pos['brain_z'][int(k)] = 0
                        if int(vix) not in vixel['brain_vixel']:
                            vixel['brain_vixel'][int(vix)] = 0
                        pos['brain_x'][int(i)] += 1
                        pos['brain_y'][int(j)] += 1
                        pos['brain_z'][int(k)] += 1
                        vixel['brain_vixel'][int(vix)] += 1
                        if i<cood_x_l:
                            cood_x_l = i
                        elif i>cood_x_r:
                            cood_x_r = i
                        if j<cood_y_l:
                            cood_y_l = j
                        elif j>cood_y_r:
                            cood_y_r = j
                        if k<cood_z_l:
                            cood_z_l = k
                        elif k>cood_z_r:
                            cood_z_r = k

        effec_range[l_f[12:]] = [(cood_x_l,cood_x_r),(cood_y_l,cood_y_r),(cood_z_l,cood_z_r)]
        if cood_x_r-cood_x_l>max_size_x:
            max_size_x = cood_x_r-cood_x_l
        if cood_y_r - cood_y_l > max_size_y:
            max_size_y = cood_y_r - cood_y_l
        if cood_z_r - cood_z_l > max_size_z:
            max_size_z = cood_z_r - cood_z_l

        shift_x_l = hippo_x_l-cood_x_l
        shift_x_r = cood_x_r-hippo_x_r
        shift_y_l = hippo_y_l-cood_y_l
        shift_y_r = cood_y_r-hippo_y_r
        shift_z_l = hippo_z_l-cood_x_l
        shift_z_r = cood_z_r-hippo_z_r
        if shift_x_l < min_shift_x_l:
            min_shift_x_l = shift_x_l
        if shift_x_r < min_shift_x_r:
            min_shift_x_r = shift_x_r
        if shift_y_l < min_shift_y_l:
            min_shift_y_l = shift_y_l
        if shift_y_r < min_shift_y_r:
            min_shift_y_r = shift_y_r
        if shift_z_l < min_shift_z_l:
            min_shift_z_l = shift_z_l
        if shift_z_r < min_shift_z_r:
            min_shift_z_r = shift_z_r

    print('s_x_l:',min_shift_x_l,' r:',min_shift_x_r)
    print('s_y_l:', min_shift_y_l, ' r:', min_shift_y_r)
    print('s_z_l:', min_shift_z_l, ' r:', min_shift_z_r)
    print('x:',max_size_x,'  y:',max_size_y,'  z:',max_size_z)

    return pos,vixel,effec_range

def EDA(labels,images,id):


    effec_range = {}
    pos = {
        'x1':{},
        'y1':{},
        'z1':{},
        'x2': {},
        'y2': {},
        'z2': {},
        'brain_x':{},
        'brain_y':{},
        'brain_z':{},
    }
    vixel = {
        'voxel1':{},
        'voxel2':{},
        'brain_voxel': {},
    }

    results = []
    pool = mlp.Pool(mlp.cpu_count())
    aver_t = int(len(labels) / mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(image_anaylizer,
                                  args=(images[i * aver_t:(i + 1) * aver_t],
                                        labels[i * aver_t:(i + 1) * aver_t]))
        results.append(result)
    pool.close()
    pool.join()

    for result in results:
        pos_i,vixel_i,effec_range_i = result.get()
        for key in pos.keys():
            for i in pos_i[key].keys():
                if i not in pos[key]:
                    pos[key][i] = 0
                pos[key][i] +=pos_i[key][i]
        for key in vixel.keys():
            for i in vixel_i[key].keys():
                if i not in vixel[key]:
                    vixel[key][i] = 0
                vixel[key][i] +=vixel_i[key][i]
        effec_range.update(effec_range_i)


    for key in pos.keys():
        pos[key]=sorted(pos[key].items(),key=lambda x:x[0],reverse = True)
    for key in vixel.keys():
        vixel[key]=sorted(vixel[key].items(),key=lambda x:x[0],reverse = True)

    with open(id+'pos.json','w') as f:
        f.write(json.dumps(pos,indent=4, separators=(',', ': ')))
    with open(id+'voxel.json','w') as f:
        f.write(json.dumps(vixel,indent=4, separators=(',', ': ')))
    with open(id+'effec_range.json','w') as f:
        f.write(json.dumps(effec_range,indent=4, separators=(',', ': ')))

def EDA_warp():
    files = get_files(PRE_IMAGE_PATH, prefix=False)

    labels = []
    images = []
    for i in range(len(files)):
        im = load_image(PRE_IMAGE_PATH + files[i]).get_data()
        if im.shape == (256, 256, 180, 1):
            labels.append(PRE_LABEL_PATH + files[i])
            images.append(PRE_IMAGE_PATH + files[i])

    EDA(labels, images, '180')

    labels = []
    images = []
    for i in range(len(files)):
        im = load_image(PRE_IMAGE_PATH + files[i]).get_data()
        if im.shape == (256, 256, 166, 1):
            labels.append(PRE_LABEL_PATH + files[i])
            images.append(PRE_IMAGE_PATH + files[i])
    EDA(labels, images, '256_166')

    labels = []
    images = []
    for i in range(len(files)):
        im = load_image(PRE_IMAGE_PATH + files[i]).get_data()
        if im.shape == (192, 192, 160, 1):
            labels.append(PRE_LABEL_PATH + files[i])
            images.append(PRE_IMAGE_PATH + files[i])
    EDA(labels, images, '192')

def get_images_info():

    def get_info(image,label):

        def fune_range(info,target,value):
            for i,t in enumerate(['x','y','z','voxel']):
                if value[i]<info[target+'_'+t][0]:
                    info[target + '_' + t][0] = value[i]

                if value[i]>info[target+'_'+t][1]:
                    info[target + '_' + t][1] = value[i]
            return info
        info = {
            'brain_x': [256,0],
            'brain_y': [256,0],
            'brain_z': [256,0],
            'brain_voxel': [10000,-10000],
            'h1_x': [256,0],
            'h1_y': [256,0],
            'h1_z': [256,0],
            'h1_voxel': [10000, -10000],
            'h2_x': [256, 0],
            'h2_y': [256, 0],
            'h2_z': [256, 0],
            'h2_voxel': [10000, -10000],
        }
        info['shape'] = image.shape
        shape = image.shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    value = (i,j,k,int(image[i,j,k,0]))
                    if label[i,j,k,0] == 1:
                        info = fune_range(info,'h1',value)
                    elif label[i,j,k,0] == 2:
                        info = fune_range(info, 'h2', value)
                    elif image[i,j,k,0] != -1:
                        info = fune_range(info, 'brain', value)
        return info

    images_info = {}
    files = get_files(PRE_IMAGE_PATH, prefix=False)

    image_files = [PRE_IMAGE_PATH+f for f in files]
    label_files = [PRE_LABEL_PATH+f for f in files]

    for l_f, im_f ,name in tqdm(zip(label_files, image_files,files)):
        # label, image = load_image(l_f).get_data(), load_image(im_f).get_data()
        label,image = sitk.ReadImage(l_f) , sitk.ReadImage(im_f)
        label,image = sitk.GetArrayFromImage(label),sitk.GetArrayFromImage(image)
        label,image = label.transpose((1,2,0)),image.transpose((1,2,0))
        label,image = label[:,:,:,np.newaxis],image[:,:,:,np.newaxis]

        images_info[name] = get_info(image,label)

    variables = [
        'brain_x',
        'brain_y',
        'brain_z',
        'brain_voxel',
        'h1_x',
        'h1_y',
        'h1_z',
        'h1_voxel',
        'h2_x',
        'h2_y',
        'h2_z',
        'h2_voxel',
    ]
    for file in images_info.keys():
        for vari in variables:
            images_info[file][vari+'_range'] = \
                images_info[file][vari][1] - images_info[file][vari][0]
    for file in images_info.keys():
        images_info[file]['h_z_range'] = \
            images_info[file]['h2_z'][1] - images_info[file]['h1_z'][0]


    for file in images_info.keys():
        for axis in ['x','y','z']:
            for h in ['h1_','h2_']:
                images_info[file]['effec_'+h+axis] = [0,0]
                images_info[file]['effec_'+h+axis][0] = images_info[file][h+axis][0]-images_info[file]['brain_'+axis][0]
                images_info[file]['effec_'+h+axis][1] = images_info[file][h+axis][1]-images_info[file]['brain_'+axis][0]

    with open(INFO+'image_info.json','w') as f:
        f.write(json.dumps(images_info,indent=4, separators=(',', ': ')))

def get_range(images_info):
    variables = ['h1_voxel','h2_voxel']

    for vari in variables:
        low_bundary = []
        high_bundary = []
        max_bundary = []
        for file in tqdm(images_info.keys()):
            info = images_info[file]
            image = load_image(PRE_IMAGE_PATH+file).get_data()
            image = np.log(image[image>=0]+1)

            # q3 = np.percentile(image,75)
            # q1 = np.percentile(image,25)
            # voxel_max = q3+2*(q3-q1)
            # voxel_min = 0
            voxel_mean = image.mean()
            voxel_std = image.std()
            voxel_max = voxel_mean + 3.5 * voxel_std
            voxel_min = voxel_mean - 3.5 * voxel_std
            voxel_range = voxel_max-voxel_min

            low = np.log(info[vari][0]+1) - voxel_min
            low_bundary.append(low/voxel_range)
            high = np.log(info[vari][1]+1) - voxel_min
            high_bundary.append(high/voxel_range)



        low_bundary = sorted(low_bundary)
        high_bundary = sorted(high_bundary)
        print(low_bundary)
        print(high_bundary)



def test():
    files = get_files(PRE_LABEL_PATH)
    h1 = []
    h2 = []
    for f in files:
        label = load_image(f).get_data()
        h1.append(np.sum(label==1))
        h2.append(np.sum(label==2))
    print(sorted(h1))
    print(sorted(h2))

if __name__=='__main__':
    test()

    # get_range(info)
    # test()
    # get_images_info()


