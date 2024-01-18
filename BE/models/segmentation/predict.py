from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from models.unets import Unet2D
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D

from utils.learning.metrics import dice_coef, precision, recall


# settings 
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
path = './data/wound_dataset/'
weight_file_name = '2019-12-19 01%3A53%3A15.480800.hdf5'
pred_save_path = '2022-11-28/'

# data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
# x_test, test_label_filenames_list = load_test_images(path)

# ### get mobilenetv2 model
model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
model = load_model('./training_history/' + weight_file_name
               , custom_objects={'recall':recall,
                                 'precision':precision,
                                 'dice_coef': dice_coef,
                                 'relu6':relu6,
                                 'DepthwiseConv2D':DepthwiseConv2D,
                                 'BilinearUpsampling':BilinearUpsampling})

# for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True): 
#     print("IMAGE BATCH SIZE = ", len(image_batch))
#     prediction = model.predict(image_batch, verbose=1)
#     save_results(prediction, 'rgb', path + 'test/predictions/' + pred_save_path, test_label_filenames_list, path + 'test/images/')
#     break
