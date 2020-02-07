import keras
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras import preprocessing
import sys
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sequence", help="The amino acid sequence for the protein being analyzed")
parser.add_argument("--dir", default="/vol/ml/pchlenski/models/function/models/", help="Directory containing files named model.h5, encoder.p, tokenizer.p")
args = parser.parse_args()

seq = args.sequence
prefix = args.dir

my_model = load_model(prefix+'model.h5')
tokenizer = pickle.load(open(prefix+"tokenizer.p", "rb"))
encoder = pickle.load(open(prefix+"encoder.p", "rb"))

encoded_seq = tokenizer.texts_to_sequences([seq])
encoded_seq = preprocessing.sequence.pad_sequences(encoded_seq, 3000)
output = my_model.predict(encoded_seq)

print(encoder.classes_[np.argmax(output)])
