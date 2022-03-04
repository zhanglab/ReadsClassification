import tensorflow as tf
import glob

print(tf.executing_eagerly())

def main():
    input_dir = sys.argv[1]
    list_ds_tensors_pred = sorted(glob.glob(os.path.join(input_dir, '*-pred-tensors')))
    for i in range(len(list_ds_tensors_pred)):
        print(i, list_ds_tensors_pred[i])
        ds_tensors_pred = tf.data.experimental.load(list_ds_tensors_pred[i])
        for elem in ds_tensors_pred:
            print(elem)
            print(type(elem))
            print(elem.numpy())
        break


if __name__ == "__main__":
    main()
