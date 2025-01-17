# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import pickle
def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            # Get the paths for the corresponding images

            #vinayak =  ['datasets/kar_Vin_aligned/vinayak/' + f for f in os.listdir('datasets/kar_Vin_aligned/vinayak')]
            #karthik =  ['datasets/kar_Vin_aligned/karthik/' + f for f in os.listdir('datasets/kar_Vin_aligned/karthik')]
            #ashish = ['datasets/kar_Vin_aligned/Ashish/' + f for f in os.listdir('datasets/kar_Vin_aligned/Ashish')]
            #saurabh = ['datasets/kar_Vin_aligned/Saurabh/' + f for f in os.listdir('datasets/kar_Vin_aligned/Saurabh')]
            #hari = ['datasets/kar_Vin_aligned/Hari/' + f for f in os.listdir('datasets/kar_Vin_aligned/Hari')]
            #paths = vinayak+karthik+ashish+saurabh+hari
            facedir = args.dir
            print("dir: "+facedir)
            names = [os.path.join(facedir, name) for name in os.listdir(facedir)
                if os.path.isdir(os.path.join(facedir, name))]
            paths = []
            for name_dir in names:
                p = [os.path.join(name_dir, f) for f in os.listdir(name_dir)]
                for f in p:
                    print("file: " + f)
                paths += p
            #np.save("images.npy",paths)
            # Load the model
            facenet.load_model(args.model)
            
            image_size = args.image_size
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            images_placeholder = tf.image.resize_images(images_placeholder,(image_size, image_size))
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            embedding_size = embeddings.get_shape()[1]
            extracted_dict = {}
            
            # Run forward pass to calculate embeddings
            for i, filename in enumerate(paths):
                if os.path.basename(filename).startswith('.'):
                    # skip files starting with dot
                    continue
                print(os.path.dirname(filename))
                images = facenet.load_image(filename, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                extracted_dict[filename] =  feature_vector
                if(i%100 == 0):
                    print("completed",i," images")

            with open('extracted_dict.pickle','wb') as f:
        	    pickle.dump(extracted_dict,f)

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--model', type=str,default='/home/gpuuser/vinayak/models/facenet/src/ckpt/20170512-110547', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--dir', type=str,
        help='Directory for aligned face images.', default='/tmp/facex')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
