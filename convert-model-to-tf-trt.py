import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from models import AlexNet
import os
import json
import sys


def main():
    ckpt = sys.argv[1]
    epoch = sys.argv[2]
    converter_mode = str(sys.argv[3])
    class_mapping = sys.argv[4]
    output_dir = sys.argv[5]

    VECTOR_SIZE = 250 - 12 + 1
    VOCAB_SIZE = 8390657
    EMBEDDING_SIZE = 60
    DROPOUT_RATE = 0.7

    f = open(class_mapping)
    class_mapping = json.load(f)
    NUM_CLASSES = len(class_mapping)

    init_lr = 0.0001
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = AlexNet(VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE)
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
    checkpoint.restore(os.path.join(ckpt, f'ckpts-{epoch}')).expect_partial()

    out_model = os.path.join(output_dir, f'ckpts-{epoch}-{converter_mode}')

    prec_modes = {'fp32': trt.TrtPrecisionMode.FP32, 'fp16': trt.TrtPrecisionMode.FP16, 'int8': trt.TrtPrecisionMode.INT8}

    # conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=prec_modes[converter_mode], max_workspace_size_bytes=40000000000)
    # converter = trt.TrtGraphConverterV2(input_saved_model_dir=model, conversion_params=conversion_params)
    # converter.convert()
    # converter.save(output_saved_model_dir=out_model)


if __name__ == "__main__":
    main()
