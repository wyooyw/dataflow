from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.dropout_attrs import *
from compiler.generate.op.tensors.dropout_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualDropout(Dual):
    def __init__(self,in_shape):
        in_batch,in_features = in_shape
        #定义张量
        mask = MemoryManager().allocActivation(shape=in_shape)
        input = MemoryManager().allocActivation(shape=in_shape) #由上一个算子申请？
        input_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        output = MemoryManager().allocActivation(shape=in_shape)
        output_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        
        forward_dropout_tensors = ForwardDropoutTensors(mask=mask,
                                                input=input,
                                                output=output)
        backward_dropout_tensors = BackwardDropoutTensors(mask=mask,
                                                input_grad=input_grad,
                                                output_grad=output_grad)

        forward_dropout_attrs = ForwardDropoutAttrs()
        backward_dropout_attrs = BackwardDropoutAttrs()
        #定义op
        self.forward = ForwardDropout(attrs=forward_dropout_attrs,
                                        tensors=forward_dropout_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardDropout(attrs=backward_dropout_attrs,
                                        tensors=backward_dropout_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
    
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualDropout(in_shape=in_shape)
        return dual


class ForwardDropout(Operator):
    def __init__(self,attrs:ForwardDropoutAttrs,tensors:ForwardDropoutTensors,in_shape=[],out_shape=[]):
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
        

class BackwardDropout(Operator):
    def __init__(self,attrs:BackwardDropoutAttrs,tensors:BackwardDropoutTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

