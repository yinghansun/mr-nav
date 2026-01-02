from typing import Tuple

import torch
import torch.onnx


def export_onnx_model(
    desired_onnx_path: str,
    model: torch.nn.Module,
    input_dim: Tuple[int],
) -> None:
    model.eval()

    dummy_input = torch.randn(input_dim)  # (batch size, input dim)

    torch.onnx.export(
        model,               # 要导出的模型
        dummy_input,         # 模型的输入
        desired_onnx_path,   # 保存的ONNX模型路径
        export_params=True,  # 存储训练好的参数权重
        opset_version=11,    # ONNX版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],   # 输入节点的名称
        output_names=['output'], # 输出节点的名称
        dynamic_axes={
            'input' : {0 : 'batch_size'},    # 批次维度动态
            'output' : {0 : 'batch_size'}
        }
    )