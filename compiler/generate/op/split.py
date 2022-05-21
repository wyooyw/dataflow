from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.split_attrs import *
from compiler.generate.op.tensors.split_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualSplit(Dual):
    def __init__(self,in_shape):
        in_batch,in_channels,in_height,in_width = in_shape


        #定义attrs
        forward_split_attrs = ForwardSplitAttrs()
        backward_split_attrs = BackwardSplitAttrs()

        #定义张量
        input = MemoryManager().allocActivation(shape=in_shape)
        input_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        output_grad1 = MemoryManager().allocFeatureGrad(shape=in_shape)
        output_grad2 = MemoryManager().allocFeatureGrad(shape=in_shape)

        forward_split_tensors = ForwardSplitTensors(input=input)
        backward_split_tensors = BackwardSplitTensors(input_grad=input_grad,
                                                output_grad1=output_grad1,
                                                output_grad2=output_grad2)
        
        #定义op
        self.forward = ForwardSplit(attrs=forward_split_attrs,
                                        tensors=forward_split_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardSplit(attrs=backward_split_attrs,
                                        tensors=backward_split_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)


class ForwardSplit(Operator):
    def __init__(self,attrs:ForwardSplitAttrs,tensors:ForwardSplitTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr:ForwardSplitAttrs):
        """根据输入shape和算子attr,计算输出shape
        """
        return in_shape

class BackwardSplit(Operator):
    def __init__(self,attrs:BackwardSplitAttrs,tensors:BackwardSplitTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)