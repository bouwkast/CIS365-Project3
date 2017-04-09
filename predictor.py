from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np


# TODO - cleanup variable names
from scipy.misc import imresize

#  names_102 is for the model with 102 classes
names_102 = {'spring_crocus': 84, 'gazania': 40, 'artichoke': 2, 'desert_rose': 32, 'mallow': 56, 'canna_lily': 20,
             'toad_lily': 92, 'wallflower': 96, 'balloon_flower': 5, 'thorn_apple': 90, 'lotus': 53,
             'prince_of_wales_feathers': 75, 'bird_of_paradise': 9, 'cape_flower': 22, 'moon_orchid': 61,
             'wild_pansy': 99,
             'mexican_aster': 58, 'foxglove': 35, 'alpine_sea_holly': 0, 'trumpet_creeper': 95, 'corn_poppy': 29,
             'magnolia': 55, 'rose': 78, 'bougainvillea': 15, 'passion_flower': 66, 'californian_poppy': 18,
             'hard_leaved_pocket_orchid': 47, 'pink_primrose': 71, 'red_ginger': 77, 'blanket_flower': 13,
             'ruby_lipped_cattleya': 79, 'sword_lily': 89, 'king_protea': 51, 'ball_moss': 4, 'pink_yellow_dahlia': 72,
             'barbeton_daisy': 6, 'lenten_rose': 52, 'morning_glory': 62, 'buttercup': 17, 'mexican_petunia': 59,
             'cautleya_spicata': 24, 'water_lily': 97, 'blackberry_lily': 12, 'poinsettia': 73, 'fire_lily': 34,
             'geranium': 41, 'tiger_lily': 91, 'clematis': 25, 'osteospermum': 64, 'sweet_william': 88,
             'globe_thistle': 44,
             'anthurium': 1, 'pelargonium': 67, 'bromelia': 16, 'purple_coneflower': 76, 'hibiscus': 48,
             'peruvian_lily': 68, 'garden_phlox': 38, 'hippeastrum': 49, 'globe_flower': 43, 'orange_dahlia': 63,
             'colts_foot': 26, 'great_masterwort': 46, 'petunia': 69, 'carnation': 23, 'tree_mallow': 93,
             'grape_hyacinth': 45, 'watercress': 98, 'gaura': 39, 'windflower': 100, 'canterbury_bells': 21,
             'siam_tulip': 80, 'fritillary': 37, 'english_marigold': 33, 'sweet_pea': 87, 'sunflower': 86,
             'common_dandelion': 28, 'monkshood': 60, 'snapdragon': 82, 'camellia': 19, 'frangipani': 36,
             'bearded_iris': 7,
             'columbine': 27, 'yello_iris': 101, 'japanese_anemone': 50, 'giant_white_arum_lily': 42, 'cyclamen': 30,
             'bee_balm': 8, 'bolero_deep_blue': 14, 'pincushion_flower': 70, 'bishop_of_llandaff': 10, 'tree_poppy': 94,
             'black_eyed_susan': 11, 'primula': 74, 'oxeye_daisy': 65, 'love_in_the_mist': 54, 'spear_thistle': 83,
             'silverbush': 81, 'daffodil': 31, 'marigold': 57, 'stemless_gentian': 85, 'azalea': 3}

names = {'buttercup': 1, 'tigerlily': 14, 'bluebell': 0, 'crocus': 4, 'daisy': 6, 'snowdrop': 12, 'lily_valley': 10,
         'tulip': 15, 'daffodil': 5, 'iris': 9, 'pansy': 11, 'colts_foot': 2, 'fritillary': 8, 'dandelion': 7,
         'cowslip': 3, 'windflower': 16, 'sunflower': 13}


def create_percentages(probabilities):
    """
    Take a numpy array containing the probabilities of some other input
    data for what appropriate flower class it belongs to.
    :param probabilities: a numpy array of float values which are probabilities
    :return: a numpy array of float values as percentages
    """
    sum = np.sum(probabilities)
    percentages = []  # standard python list to contain the percentages

    # to calculate the percentage take each independent probability and divide it by the sum of all
    for prob in np.nditer(probabilities):
        percentages.append((prob / sum) * 100)

    return percentages


def get_name(names, location):
    for name in names:
        if names[name] == location:
            return name
    return 'invalid location passed to get_name'


def top_five(percentages, names):
    five = []
    loc = 0
    for percent in percentages:
        if len(five) > 0:
            for value in five:
                if percent > value[0]:
                    five.remove(value)
                    five.append((percent, get_name(names, loc)))
                    break
                elif len(five) < 5:
                    five.append((percent, get_name(names, loc)))
                    break

        else:
            five.append((percent, get_name(names, loc)))
        loc += 1
    five.sort(key=lambda flow_tup: flow_tup[0], reverse=True)
    return five


def format_top_five(five):
    result = '\n***** Top Five Predictions *****\n\n'
    result += 'Confidence\t\tFlower Name\n'
    result += '==================================\n\n'
    for pair in five:
        result += str(round(pair[0], 2)) + '%' + '\t\t\t' + pair[1] + '\n'
    return result

def init_model(to_load=None, is_17=True):

    if to_load is None:
        print('Loading default (InceptionV3) 17 flower classifier.\n\n')
        model = load_model('model_70percent.h5')  # TODO - update to pretrained as default
    else:
        print('Loading model', to_load, '\n\n')
        model = load_model(to_load)

    if is_17:
        dict = names
    else:
        dict = names_102

    print('Model loaded successfully.\n')
    return model, dict


def predict(model_tuple):
    # TODO - dim should be 299 - need to update all models to be this size
    classes = model_tuple[1]
    model = model_tuple[0]
    dim = 299  # image height and width dimensions
    default_path = 'predict/to_predict/'
    print('Please enter the full name of the image to predict.\nNote: must be in directory predict/to_predict/\n')
    image_name = input()  # grab user input for the path of image to predict
    if image_name == 'exit':
        exit(0)
    img = load_img(default_path + image_name)
    x = imresize(img_to_array(img), (dim, dim, 3))
    x = x.reshape((1,) + x.shape)  # get the proper # of dimensions - only predict 1 image at a time

    predict_d = ImageDataGenerator(rescale=1. / 255)

    batch_size = 1

    #  TODO - the dimensions get changed here too - is this where we can get (None, dim, dim, 3)???
    for batch in predict_d.flow(x, batch_size=1):
        x = batch  # this runs in an infinite loop - need to stop it
        break

    prediction = model.predict(x, 1)

    percentages = create_percentages(prediction)
    # TODO going to hardcode to use the 17 class dict need to change!!
    print(format_top_five(top_five(percentages, classes)))



model = init_model(to_load='102_model3.h5', is_17=False)

while True:
    predict(model)

# TODO - attempt to keep model loaded in memory and run multiple predictions sequentially

# TODO - comments for the above functions

# TODO - potentially add ability to choose what model to predict on (ie. pretrained or scratch)

# TODO - cleanup prediction code - ie(split into functions)


# TODO - do we put this load_model() in a while loop to keep it loaded the whole time??
# model = load_model('model_70percent.h5')
# dim = 150
#
#
# img = load_img('predict/to_predict/image_1827263.jpg')
#
# print(img_to_array(img).shape)
#
# x = img_to_array(img)
#
# x = imresize(x, (dim, dim, 3))  # resize it to the correct dimensions
#
#
# #  TODO - need to reshape the image to have the following dimensions (None, dim, dim, 3)
#
#
#
# x = x.reshape((1,) + x.shape)
#
# # x = x.reshape((1, ) + (dim, dim, 3))
#
# predict_d = ImageDataGenerator(rescale=1. / 255)
#
# batch_size = 1
#
# #  TODO - the dimensions get changed here too - is this where we can get (None, dim, dim, 3)???
# for batch in predict_d.flow(x, batch_size=1):
#     x = batch  # this runs in an infinite loop - need to stop it
#     break
#
# # TODO - to keep model loaded we might have to change this - not sure
# predict_datagen = ImageDataGenerator(rescale=1. / 255)
#
# predict_generator = predict_datagen.flow_from_directory(
#     'predict',
#     target_size=(dim, dim),
#     batch_size=1,
#     class_mode=None)
#
# # prediction = model.predict_generator(predict_generator, 1)
# prediction = model.predict(x, 1)
#
#
#
# percentages = create_percentages(prediction)
#
#
# print(format_top_five(top_five(percentages, names)))
