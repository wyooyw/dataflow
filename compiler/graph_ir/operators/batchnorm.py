from compiler.graph_ir import Dual,Operator,OperatorType,Attrs
from compiler.graph_ir.attrs.batchnorm_attrs import *
from compiler.graph_ir.tensors.batchnorm_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn
import torch
from simulator.memory import Memory

class DualBatchnorm(Dual):
    def __init__(self,in_shape,affine):
        super().__init__()
        in_batch,in_channels,in_height,in_width = in_shape
        #定义张量
        input = MemoryManager().allocActivation(shape=in_shape)
        input_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        output = MemoryManager().allocActivation(shape=in_shape)
        output_grad = MemoryManager().allocFeatureGrad(shape=in_shape)

        avg = MemoryManager().allocActivation(shape=(in_channels,))
        std = MemoryManager().allocActivation(shape=(in_channels,))
        alpha = MemoryManager().allocWeight(shape=(in_channels,))
        beta = MemoryManager().allocWeight(shape=(in_channels,))
        
        forward_tensors = ForwardBatchnormTensors(avg=avg,
                                                std=std,
                                                alpha=alpha,
                                                beta=beta,
                                                input=input,
                                                output=output)
        backward_tensors = BackwardBatchnormTensors(avg=avg,
                                                std=std,
                                                alpha=alpha,
                                                beta=beta,
                                                input_grad=input_grad,
                                                output_grad=output_grad)

        forward_attrs = ForwardBatchnormAttrs(num_features=in_channels,affine=affine)
        backward_attrs = BackwardBatchnormAttrs(num_features=in_channels,affine=affine)
        #定义op
        self.forward = ForwardBatchnorm(attrs=forward_attrs,
                                        tensors=forward_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardBatchnorm(attrs=backward_attrs,
                                        tensors=backward_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
    
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualBatchnorm(in_shape=in_shape,affine=module.affine)
        
        # var和mean
        assert not (module.running_mean==None or module.running_var==None),"PyTorch BatchNorm2d doesn't have running_mean or running_var."
        dual.forward.get_tensors().tensors["avg"].storage.data = module.running_mean.detach()
        #TODO 改一下std的名字，改成var
        dual.forward.get_tensors().tensors["std"].storage.data = module.running_var.detach()

        # weight和bias
        if dual.forward.attrs.get("affine")==True:
            assert not (module.weight==None or module.bias==None),"PyTorch BatchNorm2d doesn't have weight or bias."
            dual.forward.get_tensors().tensors["alpha"].storage.data = module.weight.detach()
            dual.forward.get_tensors().tensors["beta"].storage.data = module.bias.detach()
        
        
        
        return dual


class ForwardBatchnorm(Operator):
    def __init__(self,attrs:ForwardBatchnormAttrs,tensors:ForwardBatchnormTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    @classmethod
    def get_in_shape_by_out_shape(cls,out_shape,attr):
        return out_shape
    
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr):
        return in_shape

    
        

class BackwardBatchnorm(Operator):
    def __init__(self,attrs:BackwardBatchnormAttrs,tensors:BackwardBatchnormTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
