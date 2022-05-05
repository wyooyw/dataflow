from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.flatten_attrs import *
from compiler.generate.op.tensors.flatten_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn


class DualFlatten(Dual):
    def __init__(self,in_shape):
        in_batch,in_channel,in_height,in_width = in_shape
        out_shape = ForwardFlatten.get_out_shape_by_in_shape(in_shape,None)
        #定义张量
        input = MemoryManager().allocActivation(shape=(in_batch,in_channel,in_height,in_width))
        input_grad = MemoryManager().allocActivation(shape=(in_batch,in_channel,in_height,in_width))
        output = MemoryManager().allocActivation(shape=(in_batch,in_channel*in_height*in_width))
        output_grad = MemoryManager().allocActivation(shape=(in_batch,in_channel*in_height*in_width))
        #定义tensors
        forward_flatten_tensors = ForwardFlattenTensors(input=input,
                                                        output=output)
        backward_flatten_tensors = BackwardFlattenTensors(output_grad=output_grad,
                                                        input_grad=input_grad)
        #attrs
        forward_flatten_attrs = ForwardFlattenAttrs()
        backward_flatten_attrs = BackwardFlattenAttrs()

        #ops
        self.forward = ForwardFlatten(attrs=forward_flatten_attrs,
                                        tensors=forward_flatten_tensors,
                                        in_shape=in_shape,
                                        out_shape=out_shape)
        self.backward = BackwardFlatten(attrs=backward_flatten_attrs,
                                        tensors=backward_flatten_tensors,
                                        in_shape=out_shape,
                                        out_shape=in_shape)
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualFlatten(in_shape=in_shape)
        return dual


class ForwardFlatten(Operator):
    def __init__(self,attrs:ForwardFlattenAttrs,tensors:ForwardFlattenTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD_FLATTEN,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr):
        """根据输入shape和算子attr,计算输出shape
        """
        assert type(in_shape)==tuple or type(in_shape)==list,f"Type if in_shape should be tuple or list, but got {in_shape}"
        assert len(in_shape)==4
        batch,channel,height,width = in_shape
        return (batch,channel * height * width)
        

class BackwardFlatten(Operator):
    def __init__(self,attrs:BackwardFlattenAttrs,tensors:BackwardFlattenTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD_FLATTEN,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

