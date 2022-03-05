import tensorflow as tf
import glob
import shutil
import sys
import os
import pickle

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

def main():
    input_dir = sys.argv[1]
    list_ds_tensors_pred = sorted(glob.glob(os.path.join(input_dir, '*-pred-tensors')))
    for i in range(len(list_ds_tensors_pred)):
        print(i, list_ds_tensors_pred[i])
        with open(os.path.join(input_dir, list_ds_tensors_pred[i], 'element_spec'), 'rb') as in_:
            es = pickle.load(in_)
        ds_tensors_pred = tf.data.experimental.load(list_ds_tensors_pred[i], es, compression='GZIP')
        for elem in ds_tensors_pred:
            print(elem)
            print(type(elem))
            print(elem.numpy())
        break


if __name__ == "__main__":
    main()
