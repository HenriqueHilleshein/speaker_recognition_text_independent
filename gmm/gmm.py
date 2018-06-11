# -*- coding: utf-8 -*-
import numpy
import os
from collections import defaultdict
from sklearn.mixture import GaussianMixture
# Mel coefficients path configuration
COEFFICIENTS_TRAINING_PATH = "../../ELSDSR/train/coefficients/"
COEFFICIENTS_TEST_PATH = "../../ELSDSR/test/coefficients/"

train_coefficients_dict = defaultdict(list)
train_filename_list = [];
test_dic = defaultdict(list)
speakers_id = []

for file in [doc for doc in os.listdir(COEFFICIENTS_TRAINING_PATH)
if doc.endswith(".out")]:
    train_filename_list.append(file);

for filename in sorted(train_filename_list):
    speaker = filename[:4]
    if speaker not in speakers_id:
        speakers_id.append(speaker)
    train_coefficients_dict[speaker].append(numpy.loadtxt(COEFFICIENTS_TRAINING_PATH
                                            +filename))

gmm = GaussianMixture(n_components=12, covariance_type="full", 
                      tol=0.001, reg_covar=1e-06, max_iter=100, 
                      n_init=1, init_params="kmeans", 
                      weights_init=None, means_init=None, 
                      precisions_init=None, random_state=None, 
                      warm_start=False, verbose=0, 
                      verbose_interval=10)

all_text_speech = numpy.empty((0,12));
for text_speech in train_coefficients_dict['FAML']:
    if all_text_speech.any():
        all_text_speech = numpy.concatenate((all_text_speech,text_speech), axis=0)
    else:
        all_text_speech = text_speech
gmm.fit(all_text_speech)
#gmm.fit(train_coefficients_dict['FAML'][4]);

labels1 = gmm.score(train_coefficients_dict['FDHH'][2])
labels2 = gmm.score(train_coefficients_dict['FAML'][2])
labels3 = gmm.score(train_coefficients_dict['FEAB'][2])
#fit(train_coefficients_dict['FAML'])