import argparse
import torch
import onnx
from onnx import optimizer
from torch.onnx import export
from models.H_vmunet import H_vmunet
from configs.config_setting import setting_config

try:
    import tensorrt as trt
except Exception:
    trt = None


def export_trt(checkpoint, output, mode='fp16'):
    model_cfg = setting_config.model_config
    model = H_vmunet(num_classes=model_cfg['num_classes'],
                     input_channels=model_cfg['input_channels'],
                     c_list=model_cfg['c_list'],
                     split_att=model_cfg['split_att'],
                     bridge=model_cfg['bridge'],
                     drop_path_rate=model_cfg['drop_path_rate'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    dummy = torch.randn(1, model_cfg['input_channels'], 256, 256)
    onnx_path = output + '.onnx'
    export(model, dummy, onnx_path, opset_version=17)
    onnx_model = onnx.load(onnx_path)
    passes = ["fuse_bn_into_conv", "fuse_add_bias_into_conv"]
    onnx_model = optimizer.optimize(onnx_model, passes)
    onnx.save(onnx_model, onnx_path)
    print('ONNX graph fused saved to', onnx_path)

    if trt is None:
        print('TensorRT not available. ONNX model exported.')
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    config = builder.create_builder_config()
    if mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = 1 << 28
    builder.max_batch_size = 1
    engine = builder.build_engine(network, config)
    with open(output + '.trt', 'wb') as f:
        f.write(engine.serialize())
    print('TensorRT engine saved to', output + '.trt')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['fp16', 'int8'], default='fp16')
    args = ap.parse_args()
    export_trt(args.checkpoint, args.output, mode=args.mode)
