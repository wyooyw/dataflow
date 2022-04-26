from compiler.generate.dual_generator import DualGenerator
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.relu_attrs import *
from compiler.generate.op.tensors.relu_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class ReluDualGenerator(DualGenerator):
    def __init__(self,in_batch,
                        in_width,
                        in_height,
                        in_channels):
        #定义张量
        mask = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width)) #由上一个算子申请？
        input_grad = None
        output = None
        output_grad = MemoryManager().allocGrad(shape=(in_batch,in_channels,in_height,in_width))
        
        forward_relu_tensors = ForwardReluTensors(mask=mask,
                                                input=input,
                                                output=output)
        backward_relu_tensors = BackwardReluTensors(mask=mask,
                                                input_grad=input_grad,
                                                output_grad=output_grad)

        forward_relu_attrs = ForwardReluAttrs()
        backward_relu_attrs = BackwardReluAttrs()
        #定义op
        self.forward_op = ForwardRelu(attrs=forward_relu_attrs,
                                        tensors=forward_relu_tensors)
        self.backward_op = BackwardRelu(attrs=backward_relu_attrs,
                                        tensors=backward_relu_tensors)


class ForwardRelu(Operator):
    def __init__(self,attrs:ForwardReluAttrs,tensors:ForwardReluTensors):
        super().__init__(type=OperatorType.FORWARD_RELU,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))
        

class BackwardRelu(Operator):
    def __init__(self,attrs:BackwardReluAttrs,tensors:BackwardReluTensors):
        super().__init__(type=OperatorType.BACKWARD_RELU,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

