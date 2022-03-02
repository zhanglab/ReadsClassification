import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from models import AlexNet
import os
import json
import sys


def main():
    model = sys.argv[1]
    converter_mode = str(sys.argv[2])
    output_dir = sys.argv[3]

    out_model = os.path.join(output_dir, f'model-{converter_mode}')

    prec_modes = {'fp32': trt.TrtPrecisionMode.FP32, 'fp16': trt.TrtPrecisionMode.FP16, 'int8': trt.TrtPrecisionMode.INT8}

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=prec_modes[converter_mode], max_workspace_size_bytes=40000000000)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model, conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir=out_model)
    print(f'successfully converted model to {converter_mode}')


if __name__ == "__main__":
    main()
