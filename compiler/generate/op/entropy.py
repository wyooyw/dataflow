from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.entropy_attrs import *
from compiler.generate.op.tensors.entropy_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn


class DualEntropy(Dual):
    def __init__(self,in_shape):
        in_batch,in_length = in_shape
        #定义张量
        input = MemoryManager().allocActivation(shape=(in_batch,in_length))
        input_grad = MemoryManager().allocGrad(shape=(in_batch,in_length))
        label = MemoryManager().allocActivation(shape=(in_batch,in_length))
        loss = MemoryManager().allocActivation(shape=(1,))
        #定义tensors
        forward_entropy_tensors = ForwardEntropyTensors(input=input,
                                                        label=label,
                                                        loss=loss)
        backward_entropy_tensors = BackwardEntropyTensors(input_grad=input_grad,
                                                        label=label,
                                                        loss=loss)
        #attrs
        forward_entropy_attrs = ForwardEntropyAttrs()
        backward_entropy_attrs = BackwardEntropyAttrs()

        #ops
        self.forward = ForwardEntropy(attrs=forward_entropy_attrs,
                                        tensors=forward_entropy_tensors,
                                        in_shape=in_shape,
                                        out_shape=1)
        self.backward = BackwardEntropy(attrs=backward_entropy_attrs,
                                        tensors=backward_entropy_tensors,
                                        in_shape=1,
                                        out_shape=in_shape)
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualEntropy(in_shape=in_shape)
        return dual


class ForwardEntropy(Operator):
    def __init__(self,attrs:ForwardEntropyAttrs,tensors:ForwardEntropyTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD_ENTROPY,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr):
        return [1]
        

class BackwardEntropy(Operator):
    def __init__(self,attrs:BackwardEntropyAttrs,tensors:BackwardEntropyTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD_ENTROPY,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

