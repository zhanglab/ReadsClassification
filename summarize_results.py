import tensorflow as tf
import glob
import shutil
import sys
import os
import pickle

# tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

# @tf.function
# def get_tensor_values(ds):
#     # sess = tf.compat.v1.Session()
#     # with sess.as_default():
#     for elem in ds:
#         print(elem)
#         print(type(elem))
#         # print_op = tf.print(elem, output_stream=sys.stdout)
#         print(elem.numpy())
#         # break
#         # sess.run(print_op)

def main():
    results_dir = sys.argv[1]
    list_ds_tensors_pred = sorted(glob.glob(os.path.join(input_dir, '*-pred-tensors')))

    for i in range(len(list_ds_tensors_pred)):
        print(i, list_ds_tensors_pred[i])
        with open(os.path.join(input_dir, list_ds_tensors_pred[i], 'element_spec'), 'rb') as in_:
            es = pickle.load(in_)
        ds_tensors_pred = tf.data.experimental.load(list_ds_tensors_pred[i], es, compression='GZIP')
        for element in ds_tensors_pred.as_numpy_iterator():
            print(element)


if __name__ == "__main__":
    main()
