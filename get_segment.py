from setting import crop_setting,OUTPUT,TEST_DATA,MODEL_PATH
from postpocess import average,ostu,threshold_filter
from nipy import load_image,save_image
from train import segment_model
from tool import swap_axis,get_files
from nipy.core.api import Image
from model import diceLoss

def model_predict(image,seg_model):
    h1, h2 = seg_model.inference(image)
    h1, h2 = seg_model.postPocess(h1), seg_model.postPocess(h2)
    h1 = swap_axis(h1,seg_model.modelname,axis=seg_model.axis)
    h2 = swap_axis(h2,seg_model.modelname,axis=seg_model.axis)
    return h1,h2

def seg_recovery(files,model_settings):

    def get_model(model_setting):
        modelname = model_setting['modelname']
        axis = model_setting['axis']
        loss = model_setting['loss']
        postPocess = model_setting['postPocess']
        seg_model = segment_model(valfiles=None, modelname=modelname,
                      axis=axis, metric=None, loss=loss, postPocess=postPocess)
        seg_model.load(model_setting['path'])
        return seg_model

    for f in files:
        img_3d = load_image(TEST_DATA+f)
        image = img_3d.get_data()
        result_h1 = []
        result_h2 = []
        for setting in model_settings:
            seg_model = get_model(setting)
            h1,h2 = model_predict(image,seg_model)
            result_h1.append(h1)
            result_h2.append(h2)
        h1 = average(result_h1)
        h2 = average(result_h2)

        h1 = h1.around()
        h2 = h2.around()

        shape = str(image.shape[2])
        area = crop_setting['3d'+shape]

        image[
            area['x'][0]:area['x'][1],
            area['y'][0]:area['y'][1],
            area['z1'][0]:area['z1'][1]
        ] = h1
        image[
            area['x'][0]:area['x'][1],
            area['y'][0]:area['y'][1],
            area['z2'][0]:area['z2'][1]
        ] = h2

        img = Image(image, img_3d.coordmap)
        save_image(img,OUTPUT+f)

def predict_test_data():

    model_files = get_files(MODEL_PATH,prefix=False)
    models = []
    for file_name in model_files:

        setting  = file_name.split('_')
        model_setting = {}
        model_setting['modelname'] = setting[0]
        model_setting['axis'] = None if setting[3] == 'None' else setting[3]
        model_setting['loss'] = diceLoss
        if setting[3] == 'ostu':
            model_setting['postPocess'] = ostu
        elif setting[3] == 'around':
            model_setting['postPocess'] = threshold_filter
        else:
            raise NameError("postPocess error")

        model_setting['path'] = MODEL_PATH + file_name

        models.append(model_setting)

    test_data = get_files(TEST_DATA,prefix=False)

    seg_recovery(test_data,models)


if __name__ == '__main__':

    predict_test_data()








