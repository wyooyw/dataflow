from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.maxpool_attrs import *
from compiler.generate.op.tensors.maxpool_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualMaxpool(Dual):
    def __init__(self,in_shape,
                        kernel_size,
                        padding=0,
                        stride=2):

        in_batch,in_width,in_height,in_channels = in_shape

        forward_attrs = ForwardMaxpoolAttrs(kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride)
        backward_attrs = BackwardMaxpoolAttrs(kernel_size=kernel_size,
                                                padding=padding,
                                                stride=stride)
        #定义张量
        input = MemoryManager().allocActivation(shape=in_shape)
        input_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        mask = MemoryManager().allocActivation(shape=in_shape)
        out_shape = ForwardMaxpool.get_out_shape_by_in_shape(in_shape,forward_attrs)
        output = MemoryManager().allocActivation(shape=out_shape)
        output_grad = MemoryManager().allocFeatureGrad(shape=out_shape)
        

        forward_tensors = ForwardMaxpoolTensors(input=input,
                                                    mask=mask,
                                                    output=output)
        backward_tensors = BackwardMaxpoolTensors(mask=mask,
                                                    output_grad=output_grad,
                                                    input_grad=input_grad)

        #定义op
        self.forward = ForwardMaxpool(attrs=forward_attrs,
                                        tensors=forward_tensors,
                                        in_shape=in_shape,
                                        out_shape=out_shape)
        self.backward = BackwardMaxpool(attrs=backward_attrs,
                                        tensors=backward_tensors,
                                        in_shape=out_shape,
                                        out_shape=in_shape)

    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualMaxpool(in_shape=in_shape,
                        kernel_size=module.kernel_size,
                        padding=module.padding,
                        stride=module.stride)
        return dual

class ForwardMaxpool(Operator):
    def __init__(self,attrs:ForwardMaxpoolAttrs,tensors:ForwardMaxpoolTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr:ForwardMaxpoolAttrs):
        """根据输入shape和算子attr,计算输出shape
        """
        assert type(in_shape)==tuple or type(in_shape)==list,f"Type if in_shape should be tuple or list, but got {in_shape}"
        assert len(in_shape)==4
        in_batch,in_channel,in_height,in_width = in_shape
        stride = attr.get("stride")
        kernel_size = attr.get("kernel_size")
        padding = attr.get("padding")
        
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        #宽高相等
        out_height = out_width
        out_channel = in_channel
        out_batch = in_batch

        return (out_batch,out_channel,out_height,out_width)

class BackwardMaxpool(Operator):
    def __init__(self,attrs:BackwardMaxpoolAttrs,tensors:BackwardMaxpoolTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

