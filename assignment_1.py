import numpy as np
import glob
from PIL import Image
import csv
import sys

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
np.set_printoptions(threshold=sys.maxint)

# read from query folder
image_files = []
total_images = 0
for file_name in glob.glob("INFO3406_assignment1_query/*.png"):
    image = Image.open(file_name)
    image_array = np.array(image)
    red_arr = np.array(image_array[:,0:,0]).flatten()
    green_arr = np.array(image_array[:,0:,1]).flatten()
    blue_arr = np.array(image_array[:,0:,2]).flatten()
    temp_img_arr = np.concatenate((red_arr, green_arr, blue_arr))
    image_files.append(temp_img_arr)
    total_images += 1

image_files = np.asarray(image_files) # cast to numpy nd array

data_batch_1 = unpickle('data_batch_1')
data_batch_2 = unpickle('data_batch_2')
data_batch_3 = unpickle('data_batch_3')
data_batch_4 = unpickle('data_batch_4')
data_batch_5 = unpickle('data_batch_5')
test_batch = unpickle('test_batch')
batches_meta = unpickle('batches.meta')

# data np arrays
d1 = data_batch_1['data']
d2 = data_batch_2['data']
d3 = data_batch_3['data']
d4 = data_batch_4['data']
d5 = data_batch_5['data']

# label np arrays
l1 = data_batch_1['labels']
l2 = data_batch_2['labels']
l3 = data_batch_3['labels']
l4 = data_batch_4['labels']
l5 = data_batch_5['labels']

# find the test values
test_all = test_batch['data']
test_labels = test_batch['labels']

total_images = 1000
test = test_all[0:total_images]

# test = image_files
# test_labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]

# join all test data
label = np.concatenate((l1, l2, l3, l4, l5), axis = 0)
training_data = np.vstack([d1, d2, d3, d4, d5])

# define empty classification arrays
count = np.zeros((total_images,)*2)
i = 0
for t in test:
    distance = (np.sum((training_data-t)**2, axis=1))**0.5 # get distance to all other neighbours
    for j in distance.argsort()[:111]: #change to allow user to input k
        count[i][label[j]] += 1
    i += 1

classification = []
# take the majority vote of class labels among the knn
for c in count:
    most_likely = c.argmax(axis=0)
    classification.append(most_likely)

print(classification[0:total_images:1])
print(test_labels[0:total_images:1])
n_correct = 0
for k in range(0,total_images):
    if test_labels[k] == classification[k]:
        n_correct += 1

print(n_correct)

print str(float(n_correct)/float(len(test))) #percentage correct

classification_csv = np.array(classification)
test_labels_csv = np.array(test_labels)

np.savetxt("classification.csv", classification, newline=', ', delimiter=",", fmt='%s')
np.savetxt("test_labels.csv", test_labels_csv, newline=', ', delimiter=",", fmt='%s')
