"""
predictor.py can load in one of the available models and when passed an image will predict what type of flower it is

Available models:
    102_model3.h5
            this model is a fine-tuned InceptionV3 network for the 102 flower dataset, achieved 95% accuracy
    model_98percent.h5
            this model is a fine-tuned InceptionV3 network for the 17 flower dataset, achieved 98% accuracy
    model_70percent.h5
            this model is a from scratch convolutional network for the 17 flower dataset, achived 70% accuracy
                    this reduced accuracy is because 80 images per class simply isn't enough by itself
                    but is still quite impressive

Requirements to run:
    keras, numpy, scipy - all must be latest version as of 4/11/2017
    tensorflow OR tensorflow-gpu
        tensorflow is the CPU-based version and works just fine for the predictor and is simple to install
            pip install tensorflow
        tensorflow-gpu is the GPU-based version and is much more difficult to setup, and is only beneficial
            for training the model. (Not recommended if only desire is to run predictor.py)

    All images to predict must be placed in the director structure 'predict/to_predict/' where predict is
    in the same director as the models and this predictor.py file.

To actually predict:
    run: python predictor.py
    May have to wait for tensorflow to load in
    When prompted enter in the filename of the model to use for predictions
    Answer y/n for whether it is the 102 flower dataset
    Wait about 30s or more for the model to be loaded into memory

    After the model is loaded it will prompt the user to enter the full name of the image (extension included)
        to  predict
    After entering the first image it will take about 5 seconds to output the top 5 predictions

    All subsequent images 'should' be predicted nearly instantaneously.

    To exit either to a Ctrl+c or type 'exit'

"""


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
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

#  names is for the 17 classes
names = {'buttercup': 1, 'tigerlily': 14, 'bluebell': 0, 'crocus': 4, 'daisy': 6, 'snowdrop': 12, 'lily_valley': 10,
         'tulip': 15, 'daffodil': 5, 'iris': 9, 'pansy': 11, 'colts_foot': 2, 'fritillary': 8, 'dandelion': 7,
         'cowslip': 3, 'windflower': 16, 'sunflower': 13}


def ask_for_model():
    """
    Ask the user for what model that want to load (filename) and what dataset:
    either 17 or 102
    :return: a tuple containing (filename, dataset)
        filename - string
        dataset - int
    """
    valid_files = ['102_model.h5', 'model_98percent.h5', 'model_70percent.h5']
    filename = input('Please enter the model that you want to load:\n'
                     'Available models are:\n'
                     '\t102_model.h5\n'
                     '\tmodel_98percent.h5\n'
                     '\tmodel_70percent.h5\n'
                     'Model to load:\t')
    if filename not in valid_files:
        print('Valid files are ', valid_files)
        print('Invalid filename entered. Exiting.')
        exit(1)
    is_17 = True
    if filename == '102_model.h5':
        is_17 = False  # we only have one model that is for the 102 dataset

    return filename, is_17


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
    """
    Reads in the appropriate dictionary of classes and the location of the class we want
    :param names: dictionary of classes and integer labels
    :param location: integer label of flower
    :return: the name of the flower that lines up with the passes location
    """
    for name in names:
        if names[name] == location:
            return name
    return 'invalid location passed to get_name'


def top_five(percentages, names):
    """
    Create the top 5 predictions for the given flower and convert them into percentages.
    :param percentages: list of percentages that line up with class labels
    :param names: is the dictionary that contains the class names and their integer labels
    :return: a list of the top five percentages as tuples with (percent, name_of_flower)
    """
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
    """
    Format the top five predictions into a more pleasing look
    :param five: list of top five percentages containing tuples (percentage, name)
    :return: Formatted string of the predictions
    """
    result = '\n***** Top Five Predictions *****\n\n'
    result += 'Confidence\t\tFlower Name\n'
    result += '==================================\n\n'
    for pair in five:
        result += str(round(pair[0], 2)) + '%' + '\t\t\t' + pair[1] + '\n'
    return result


def init_model(to_load=None, is_17=True):
    """
    Initialize the model to use for predicting
    :param to_load: file name of the model to load
    :param is_17: whether it is the 17 class model or the 102 class model
    :return: a tuple containing the (model, classdict)
    """
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
    """
    Prompts the user for an image to predict.

    Preprocess the image to regularize the RGB colors
            resizes the image to dimensions of (1, dim, dim, 3)
            where 1 is the batch size
                dim is the HxW of the image
                and 3 is how many color channels there are
    :param model_tuple: is a tuple that contains (model, dictionary)
    :return: The top five predictions for the given image
    """
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

    predict_d = ImageDataGenerator(rescale=1. / 255)  # regularize
    # preprocess the image in an infinite loop
    for batch in predict_d.flow(x, batch_size=1):
        x = batch  # this runs in an infinite loop - need to stop it
        break

    prediction = model.predict(x, 1)  # predict what the classes should be

    percentages = create_percentages(prediction)
    print(format_top_five(top_five(percentages, classes)))


# grab user input for what model that we should load
user_input = ask_for_model()
model = init_model(to_load=user_input[0], is_17=user_input[1])

while True:
    predict(model)
