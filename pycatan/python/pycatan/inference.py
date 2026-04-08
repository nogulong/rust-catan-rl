from __future__ import annotations
import numpy as np
import abc
import logging
from typing import Tuple

class InferenceModel(abc.ABC):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
    
    def infer():
        pass

class OnnxInfer(InferenceModel):
    first_load = True

    def __init__(self, path: str, device: str):
        import re
        super().__init__(device)
        import onnxruntime as ort
        if device == '':
            device = 'cpu'
        if device == 'cpu':
            provider = ['CPUExecutionProvider']
        elif device.startswith('cuda'):
            cuda_pattern = r'^cuda:([0-9]+)$'
            if match := re.match(cuda_pattern, device):
                provider = [('CUDAExecutionProvider',
                             {'device_id': int(match.group(1))}),
                            'CPUExecutionProvider']
            else:
                provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            provider = ort.get_available_providers()
        self.ort_session = ort.InferenceSession(path, providers=provider)
        if OnnxInfer.first_load:
            logging.info(self.ort_session.get_providers())
            OnnxInfer.first_load = False
        self.binding = self.ort_session.io_binding()
        logging.debug(f'{device=}')
        self.device = device

    def infer(self, board: np.ndarray, flat: np.ndarray):
        return self.infer_naive(board, flat)
    
    
    def infer_naive(self, board: np.ndarray, flat: np.ndarray):
        """torchを使わないコード"""
        input = {
            "board": board.astype(np.float32),
            "flat": flat.astype(np.float32)
        }
        out = self.ort_session.run(None, input)
        return out
    

class InferenceModel_TradeExpector(abc.ABC):
    def __init__(self, device: str):
        super().__init__()
        self.device = device


class OnnxInfer_TradeExpector(InferenceModel_TradeExpector):
    first_load = True

    def __init__(self, path: str, device: str):
        import re
        super().__init__(device)
        import onnxruntime as ort
        if device == '':
            device = 'cpu'
        if device == 'cpu':
            provider = ['CPUExecutionProvider']
        elif device.startswith('cuda'):
            cuda_pattern = r'^cuda:([0-9]+)$'
            if match := re.match(cuda_pattern, device):
                provider = [('CUDAExecutionProvider',
                             {'device_id': int(match.group(1))}),
                            'CPUExecutionProvider']
            else:
                provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            provider = ort.get_available_providers()
        self.ort_session = ort.InferenceSession(path, providers=provider)
        if OnnxInfer.first_load:
            logging.info(self.ort_session.get_providers())
            OnnxInfer.first_load = False
        self.binding = self.ort_session.io_binding()
        logging.debug(f'{device=}')
        self.device = device
    
    def infer_naive(self, board: np.ndarray, flat: np.ndarray):
        """torchを使わないコード"""
        input = {
            "board": board.astype(np.float32),
            "flat": flat.astype(np.float32)
        }
        out = self.ort_session.run(None, input)
        return out  