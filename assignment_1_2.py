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

train_batch = unpickle('train')
test_batch = unpickle('test')
batches_meta = unpickle('meta')

# data np arrays
train_dat = train_batch['data']
test_dat = test_batch['data']

# subclass it belongs to
train_fine_lab = train_batch['fine_labels']
test_fine_lab = test_batch['fine_labels']

# superclass it belongs to
train_coarse_lab = train_batch['coarse_labels']
test_coarse_lab = test_batch['coarse_labels']

total_images = 100
test = test_dat[0:total_images]

# define empty classification arrays
count = np.zeros((total_images,)*2)
sub_count = np.zeros((total_images,)*2)
cls = []
i = 0
#count how many classifications have occured in each
for t in test:
    distance = (np.sum((train_dat-t)**2, axis=1))**0.5
    for j in distance.argsort()[:111]:
        count[i][train_coarse_lab[j]] += 1
        sub_count[i][train_fine_lab[j]] += 1
    i += 1

# Most likely classification
classification = []
for c in count:
    most_likely = c.argmax(axis=0)
    classification.append(most_likely)

sub_classification = []
for s in sub_count:
    most_likely_sub = s.argmax(axis=0)
    sub_classification.append(most_likely_sub)

print classification[0:total_images:1]
print test_coarse_lab[0:total_images:1]
print sub_classification[0:total_images:1]
print test_fine_lab[0:total_images:1]

n_correct = 0
for k in range(0,total_images):
    if test_coarse_lab[k] == classification[k]:
        n_correct += 1

# print edmundBong
print 'percentage correct: ', float(n_correct)/float(total_images)

sub_n = 0
for k in range(0, total_images):
    if test_fine_lab[k] == sub_classification[k]:
        sub_n += 1
print 'sub class percentage correct: ', float(sub_n)/float(total_images)


np.savetxt("super_classification.csv", classification, newline=', ', delimiter=",", fmt='%s')
np.savetxt("super_test_labels.csv", test_coarse_lab, newline=', ', delimiter=",", fmt='%s')
np.savetxt("sub_classification.csv", sub_classification, newline=', ', delimiter=",", fmt='%s')
np.savetxt("sub_test_labels.csv", test_fine_lab, newline=', ', delimiter=",", fmt='%s')
