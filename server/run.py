#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                  	       
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
#from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
#from flask.ext.httpauth import HTTPBasicAuth
#from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import sys
import random
from tensorflow.python.platform import gfile
from six import iteritems
sys.path.append('..')
import numpy as np
from lib.src import retrieve
from lib.src.align import detect_face
from lib.src.facenet import load_model
import tensorflow as tf
import pickle
from tensorflow.python.platform import gfile
import argparse
#app = Flask(__name__, static_url_path = "")

#auth = HTTPBasicAuth()

#==============================================================================================================================
#                                                                                                                              
#    Loading the stored face embedding vectors for image retrieval                                                                 
#                                                                          						        
#                                                                                                                              
#==============================================================================================================================
#with open('../lib/src/face_embeddings.pickle','rb') as f:
def main(args):
    with open(args.pickle,'rb') as f:
        feature_array = pickle.load(f, encoding='utf-8') 

        #model_exp = '../lib/src/ckpt/20170512-110547'
        model_exp = args.model
        graph_fr = tf.Graph()
        sess_fr = tf.Session(graph=graph_fr)
        with graph_fr.as_default():
            with sess_fr.as_default():
                #saverf = tf.train.import_meta_graph(os.path.join(model_exp, 'model-20180408-102900.meta'))
                #saverf.restore(sess_fr, os.path.join(model_exp, 'model-20180408-102900.ckpt-90'))
                load_model(model_exp)
                pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)
                retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array)
#==============================================================================================================================
#                                                                                                                              
#  This function is used to do the face recognition from video camera                                                          
#                                                                                                 
#                                                                                                                              
#==============================================================================================================================
#@app.route('/facerecognitionLive', methods=['GET', 'POST'])
#def face_det():
#    retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array)

#==============================================================================================================================
#                                                                                                                              
#                                           Main function                                                        	            #						     									       
#  				                                                                                                
#==============================================================================================================================
#@app.route("/")
#def main():
    #return render_template("main.html")   
#if __name__ == '__main__':
#    app.run(debug = True, host= '0.0.0.0')

#face_det()
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str,
        help='Path for pickle formatted dict to use', default='../extracted_dict.pickle')
    parser.add_argument('--model', type=str,default='../20180408-102900', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
