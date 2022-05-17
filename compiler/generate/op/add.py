from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.add_attrs import *
from compiler.generate.op.tensors.add_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualAdd(Dual):
    def __init__(self,in_shape):
        in_batch,in_channels,in_height,in_width = in_shape


        #定义attrs
        forward_add_attrs = ForwardAddAttrs()
        backward_add_attrs = BackwardAddAttrs()

        #定义张量
        input1 = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input2 = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        grad = MemoryManager().allocGrad(shape=(in_batch,in_channels,in_height,in_width))

        forward_add_tensors = ForwardAddTensors(input1=input1,
                                                input2=input2,
                                                output=output)
        backward_add_tensors = BackwardAddTensors(grad=grad)
        
        #定义op
        self.forward = ForwardAdd(attrs=forward_add_attrs,
                                        tensors=forward_add_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardAdd(attrs=backward_add_attrs,
                                        tensors=backward_add_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)


class ForwardAdd(Operator):
    def __init__(self,attrs:ForwardAddAttrs,tensors:ForwardAddTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.AGGREGATE,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr:ForwardAddAttrs):
        """根据输入shape和算子attr,计算输出shape
        """
        return in_shape

class BackwardAdd(Operator):
    def __init__(self,attrs:BackwardAddAttrs,tensors:BackwardAddTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.SPLIT,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)