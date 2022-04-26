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
                        padding=0):

        #定义张量
        weight = MemoryManager().allocWeight(shape=(out_channels,in_channels,kernel_size,kernel_size))
        weight_grad = MemoryManager().allocGrad(shape=(out_channels,in_channels,kernel_size,kernel_size))
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input_grad = None
        output = None
        output_size = in_width + 2*padding - kernel_size + 1
        output_grad = MemoryManager().allocGrad(shape=(in_batch,out_channels,output_size,output_size))

        forward_conv_attrs = ForwardConvAttrs(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                padding=padding)
        backward_conv_attrs = BackwardConvAttrs(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                padding=padding)

        forward_conv_tensors = ForwardConvTensors(weight=weight,
                                                    input=input,
                                                    output=output)
        backward_conv_tensors = BackwardConvTensors(weight=weight,
                                                    input=input,
                                                    output_grad=output_grad,
                                                    weight_grad=weight_grad,
                                                    input_grad=input_grad)

        #定义op
        self.forward_op = ForwardConv2d(attrs=forward_conv_attrs,
                                        tensors=forward_conv_tensors)
        self.backward_op = BackwardConv2d(attrs=backward_conv_attrs,
                                        tensors=backward_conv_tensors)

class ForwardConv2d(Operator):
    def __init__(self,attrs:ForwardConvAttrs,tensors:ForwardConvTensors):
        super().__init__(type=OperatorType.FORWARD_CONV,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class BackwardConv2d(Operator):
    def __init__(self,attrs:BackwardConvAttrs,tensors:BackwardConvTensors):
        super().__init__(type=OperatorType.BACKWARD_CONV,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

