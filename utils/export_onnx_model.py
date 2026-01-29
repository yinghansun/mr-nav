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
        model,
        dummy_input,
        desired_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},
            'output' : {0 : 'batch_size'}
        }
    )