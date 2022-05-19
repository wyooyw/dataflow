from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.batchnorm_attrs import *
from compiler.generate.op.tensors.batchnorm_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualBatchnorm(Dual):
    def __init__(self,in_shape):
        in_batch,in_channels,in_height,in_width = in_shape
        #定义张量
        input = MemoryManager().allocActivation(shape=in_shape)
        input_grad = MemoryManager().allocGrad(shape=in_shape)
        output = MemoryManager().allocActivation(shape=in_shape)
        output_grad = MemoryManager().allocGrad(shape=in_shape)

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

        forward_attrs = ForwardBatchnormAttrs(num_features=in_channels)
        backward_attrs = BackwardBatchnormAttrs(num_features=in_channels)
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
        dual = DualBatchnorm(in_shape=in_shape)
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

