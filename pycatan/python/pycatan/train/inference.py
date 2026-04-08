import numpy as np
import abc
import logging
from typing import Tuple
from pycatan import OnnxInfer, OnnxInfer_TradeExpector
import torch

class OnnxInferTorch(OnnxInfer):
    first_load = True
    
    def infer(self, board: torch.Tensor, flat: torch.Tensor):
        return self.infer_iobinding(board.to(self.device), flat.to(self.device))
        # return self.infer_naive(inputs)
    
    def infer_iobinding(self, board, flat):
        """work with torch 2.5.1, onnxruntime-gpu 1.20.1
        """
        board = board.contiguous()
        flat = flat.contiguous()
        device = self.device
        self.binding.bind_input(
            name='board',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(board.shape), buffer_ptr=board.data_ptr(),
        )
        self.binding.bind_input(
            name='flat',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(flat.shape), buffer_ptr=flat.data_ptr(),
        )
        move_tensor = torch.empty((board.shape[0], 374),
                                  dtype=torch.float32,
                                  device=device).contiguous()
        self.binding.bind_output(
            name='move',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(move_tensor.shape),
            buffer_ptr=move_tensor.data_ptr(),
        )
        # binding.bind_output('move')
        value_tensor = torch.empty((board.shape[0], 1),
                                   dtype=torch.float32,
                                   device=device).contiguous()
        self.binding.bind_output(
            name='value',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(value_tensor.shape),
            buffer_ptr=value_tensor.data_ptr(),
        )
        # binding.bind_output('value')
        aux_tensor = torch.empty((board.shape[0],120*3),
                                 dtype=torch.float32,
                                 device=device).contiguous()
        self.binding.bind_output(
            name='aux',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(aux_tensor.shape),
            buffer_ptr=aux_tensor.data_ptr(),
        )
        # binding.bind_output('aux')
        self.ort_session.run_with_iobinding(self.binding)
        return (move_tensor.to('cpu').numpy(),
                value_tensor.to('cpu').numpy(),
                aux_tensor.to('cpu').numpy())
        # out = binding.copy_outputs_to_cpu()

class OnnxInferTorch_TradeExpector(OnnxInfer_TradeExpector):
    first_load = True
    
    def infer(self, board: torch.Tensor, flat: torch.Tensor):
        return self.infer_iobinding(board.to(self.device), flat.to(self.device))
        # return self.infer_naive(inputs)
    
    def infer_iobinding(self, board, flat):
        """work with torch 2.5.1, onnxruntime-gpu 1.20.1
        """
        board = board.contiguous()
        flat = flat.contiguous()
        device = self.device
        self.binding.bind_input(
            name='board',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(board.shape), buffer_ptr=board.data_ptr(),
        )
        self.binding.bind_input(
            name='flat',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(flat.shape), buffer_ptr=flat.data_ptr(),
        )
        prob_tensor = torch.empty((board.shape[0], 1),
                                  dtype=torch.float32,
                                  device=device).contiguous()
        self.binding.bind_output(
            name='prob',
            device_type=device, device_id=0, element_type=np.float32,
            shape=tuple(prob_tensor.shape),
            buffer_ptr=prob_tensor.data_ptr(),
        )
        # binding.bind_output('move')
        self.ort_session.run_with_iobinding(self.binding)
        return prob_tensor.to('cpu').numpy()
        # out = binding.copy_outputs_to_cpu()

        

def export_onnx(model, config, device, filename, remove_aux_head=False):
    import onnx                 # to detect import error eaelier
    onnx.__version__
    import torch.onnx
    model.eval()
    dtype = torch.float
    w, h, c = config["input_board_shape"]
    flat_dim = config["input_scalar_dim"]
    dummy_board = torch.randn(1024, c, h, w, device=device,
                              dtype=dtype)
    dummy_flat = torch.randn(1024, flat_dim, device=device, dtype=dtype)
    dummy_input = (dummy_board, dummy_flat)

    if not filename.endswith('.onnx'):
        filename = f'{filename}.onnx'

    if not remove_aux_head:
        torch.onnx.export(model, dummy_input, filename,
                          dynamic_axes={'board': {0: 'batch_size'},
                                        'flat': {0: 'batch_size'},
                                        'move': {0: 'batch_size'},
                                        'value': {0: 'batch_size'},
                                        'aux': {0: 'batch_size'}},
                          verbose=False, input_names=['board', 'flat'],
                          output_names=['move', 'value', 'aux'])
    else:
        torch.onnx.export(model, dummy_input, filename,
                          dynamic_axes={'board': {0: 'batch_size'},
                                        'flat': {0: 'batch_size'},
                                        'move': {0: 'batch_size'},
                                        'value': {0: 'batch_size'}},
                          verbose=False, input_names=['board', 'flat'],
                          output_names=['move', 'value', 'aux'])

def export_onnx_trade(model, config, device, filename):
    import onnx                 # to detect import error eaelier
    onnx.__version__
    import torch.onnx
    model.eval()
    dtype = torch.float
    w, h, c = config["input_board_shape"]
    flat_dim = config["input_scalar_dim"] + 8
    dummy_board = torch.randn(1024, c, h, w, device=device,
                              dtype=dtype)
    dummy_flat = torch.randn(1024, flat_dim, device=device, dtype=dtype)
    dummy_input = (dummy_board, dummy_flat)

    if not filename.endswith('.onnx'):
        filename = f'{filename}.onnx'

    torch.onnx.export(model, dummy_input, filename,
                        dynamic_axes={'board': {0: 'batch_size'},
                                    'flat': {0: 'batch_size'},
                                    'prob': {0: 'batch_size'}},
                        verbose=False, input_names=['board', 'flat'],
                        output_names=['prob'])
        


        