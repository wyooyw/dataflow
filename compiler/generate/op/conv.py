from compiler.generate.dual_generator import DualGenerator
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.conv_attrs import *
from compiler.generate.op.tensors.conv_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class ConvDualGenerator(DualGenerator):
    def __init__(self,in_batch,
                        in_width,
                        in_height,
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=0,
                        stride=1):


        forward_conv_attrs = ForwardConvAttrs(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride)
        backward_conv_attrs = BackwardConvAttrs(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                stride=stride)
        #定义张量
        weight = MemoryManager().allocWeight(shape=(out_channels,in_channels,kernel_size,kernel_size))
        weight_grad = MemoryManager().allocGrad(shape=(out_channels,in_channels,kernel_size,kernel_size))
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input_grad = None
        output = None
        #TODO output_size把stride考虑进来，调用ForwardConv的get_out_shape_by_in_shape
        in_shape = (in_batch,in_channel,in_height,in_width)
        out_shape = ForwardConv.get_out_shape_by_in_shape(in_shape)
        output_grad = MemoryManager().allocGrad(shape=out_shape)

        forward_conv_tensors = ForwardConvTensors(weight=weight,
                                                    input=input,
                                                    output=output)
        backward_conv_tensors = BackwardConvTensors(weight=weight,
                                                    input=input,
                                                    output_grad=output_grad,
                                                    weight_grad=weight_grad,
                                                    input_grad=input_grad)

        #定义op
        self.forward_op = ForwardConv(attrs=forward_conv_attrs,
                                        tensors=forward_conv_tensors)
        self.backward_op = BackwardConv(attrs=backward_conv_attrs,
                                        tensors=backward_conv_tensors)

class ForwardConv(Operator):
    def __init__(self,attrs:ForwardConvAttrs,tensors:ForwardConvTensors):
        super().__init__(type=OperatorType.FORWARD_CONV,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

    @classmethod
    def get_in_shape_by_out_shape(out_shape,attr:ForwardConvAttrs):
        """根据输出shape和算子attr,计算输入shape
        """
        assert type(out_shape)==tuple or type(out_shape)==list
        assert len(out_shape)==4
        out_batch,out_channel,out_height,out_width = out_shape
        in_channels = attr.get("in_channels")
        stride = attr.get("stride")
        kernel_size = attr.get("kernel_size")
        padding = attr.get("padding")
        """
        实际上,in_width的范围是[out_width*stride+kernel_size-2*padding-stride,out_width*stride+kernel_size-2*padding-1],
        这里取了下确界
        推导见https://www.wolai.com/2xMkPwJEPuiYAzuQN9fDpx
        """
        in_width = out_width * stride + kernel_size - 2 * padding - stride

        """
        宽高相等
        """
        in_height = in_width

        in_channel = in_channels

        in_batch = out_batch

        return (in_batch,in_channel,in_height,in_width)

    @classmethod
    def get_out_shape_by_in_shape(in_shape,attr:ForwardConvAttrs):
        """根据输入shape和算子attr,计算输出shape
        """
        assert type(in_shape)==tuple or type(in_shape)==list
        assert len(in_shape)==4
        in_batch,in_channel,in_height,in_width = in_shape
        out_channels = attr.get("out_channels")
        stride = attr.get("stride")
        kernel_size = attr.get("kernel_size")
        padding = attr.get("padding")
        
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1

        """
        宽高相等
        """
        out_height = out_width

        out_channel = out_channels

        out_batch = in_batch

        return (out_batch,out_channel,out_height,out_width)

class BackwardConv(Operator):
    def __init__(self,attrs:BackwardConvAttrs,tensors:BackwardConvTensors):
        super().__init__(type=OperatorType.BACKWARD_CONV,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

