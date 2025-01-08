
import torch, onnx2torch
from torch import tensor
class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        setattr(self,'Gemm', torch.nn.modules.linear.Linear(**{'in_features':312,'out_features':20}))
        setattr(self,'Gemm_1', torch.nn.modules.linear.Linear(**{'in_features':20,'out_features':128}))
        setattr(self,'Relu', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Gemm_2', torch.nn.modules.linear.Linear(**{'in_features':128,'out_features':64}))
        setattr(self,'Relu_1', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Gemm_3', torch.nn.modules.linear.Linear(**{'in_features':64,'out_features':2}))
        setattr(self,'Gemm_4', torch.nn.modules.linear.Linear(**{'in_features':2,'out_features':4}))
    def forward(self, input_1):
        gemm = self.Gemm(input_1);  input_1 = None
        gemm_1 = self.Gemm_1(gemm);  gemm = None
        relu = self.Relu(gemm_1);  gemm_1 = None
        gemm_2 = self.Gemm_2(relu);  relu = None
        relu_1 = self.Relu_1(gemm_2);  gemm_2 = None
        gemm_3 = self.Gemm_3(relu_1);  relu_1 = None
        gemm_4 = self.Gemm_4(gemm_3);  gemm_3 = None
        return gemm_4
    
