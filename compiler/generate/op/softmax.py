from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.softmax_attrs import *
from compiler.generate.op.tensors.softmax_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualSoftmax(Dual):
    def __init__(self,in_shape):
        in_batch,in_length = in_shape
        #定义张量
        input = MemoryManager().allocActivation(shape=(in_batch,in_length))
        input_grad = MemoryManager().allocActivation(shape=(in_batch,in_length))
        output = MemoryManager().allocActivation(shape=(in_batch,in_length))
        output_grad = MemoryManager().allocGrad(shape=(in_batch,in_length))
        #定义tensors
        forward_softmax_tensors = ForwardSoftmaxTensors(input=input,
                                                        output=output)
        backward_softmax_tensors = BackwardSoftmaxTensors(input_grad=input_grad,
                                                        output_grad=output_grad)
        #attrs
        forward_softmax_attrs = ForwardSoftmaxAttrs()
        backward_softmax_attrs = BackwardSoftmaxAttrs()

        #ops
        self.forward = ForwardSoftmax(attrs=forward_softmax_attrs,
                                        tensors=forward_softmax_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardSoftmax(attrs=backward_softmax_attrs,
                                        tensors=backward_softmax_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualSoftmax(in_shape=in_shape)
        return dual

class ForwardSoftmax(Operator):
    def __init__(self,attrs:ForwardSoftmaxAttrs,tensors:ForwardSoftmaxTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD_SOFTMAX,
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
        

class BackwardSoftmax(Operator):
    def __init__(self,attrs:BackwardSoftmaxAttrs,tensors:BackwardSoftmaxTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD_SOFTMAX,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)