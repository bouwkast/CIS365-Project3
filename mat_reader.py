from scipy.io import loadmat
import os

# prune the training directory to remove the first 25% of images, leaving 75% for training
for (dirpath, dirnames, filenames) in os.walk('102_flowers/train'):
    count = 0
    num_files = len(filenames)
    to_remove = len(filenames) // 4
    while count < to_remove:
        os.remove(dirpath + '/' + filenames[count])
        count += 1
# reverse the list of files, then remove the first 75% of images, leaving 25% for validation
for (dirpath, dirnames, filenames) in os.walk('102_flowers/validation'):
    count = 0
    num_files = len(filenames)
    to_remove = num_files - (num_files // 4)
    filenames.reverse()
    while count < to_remove:
        os.remove(dirpath + '/' + filenames[count])
        count += 1

imagelabels = {}

loadmat('imagelabels.mat', mdict=imagelabels)


count = 0
index = 1
locations = []
class_counts = {}
for label_arr in imagelabels['labels']:
    for label in label_arr:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
        # f.write(str(label) + ' \n')
        if label == 1:
            locations.append(index)
            count += 1
        # following two lines will move the files into directories by their labels
        image_name = str(index).zfill(5)
        os.rename('102_flowersCopy/image_' + image_name + '.jpg', '102_flowersCopy/' + str(label) + '/image_' + image_name + '.jpg')
        index += 1

