from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.relu_attrs import *
from compiler.generate.op.tensors.relu_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualRelu(Dual):
    def __init__(self,in_shape):
        # in_batch,in_channels,in_height,in_width = in_shape
        #定义张量
        mask = MemoryManager().allocActivation(shape=in_shape)
        input = MemoryManager().allocActivation(shape=in_shape) #由上一个算子申请？
        input_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        output = MemoryManager().allocActivation(shape=in_shape)
        output_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        
        forward_relu_tensors = ForwardReluTensors(mask=mask,
                                                input=input,
                                                output=output)
        backward_relu_tensors = BackwardReluTensors(mask=mask,
                                                input_grad=input_grad,
                                                output_grad=output_grad)

        forward_relu_attrs = ForwardReluAttrs()
        backward_relu_attrs = BackwardReluAttrs()
        #定义op
        self.forward = ForwardRelu(attrs=forward_relu_attrs,
                                        tensors=forward_relu_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardRelu(attrs=backward_relu_attrs,
                                        tensors=backward_relu_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
    
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualRelu(in_shape=in_shape)
        return dual


class ForwardRelu(Operator):
    def __init__(self,attrs:ForwardReluAttrs,tensors:ForwardReluTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD_RELU,
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
        

class BackwardRelu(Operator):
    def __init__(self,attrs:BackwardReluAttrs,tensors:BackwardReluTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD_RELU,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

