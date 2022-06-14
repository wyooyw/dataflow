from compiler.graph_ir import Dual,Operator,OperatorType,Attrs
from compiler.graph_ir.attrs.entropy_attrs import *
from compiler.graph_ir.tensors.entropy_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn
import torch
from simulator.memory import Memory

class DualEntropy(Dual):
    def __init__(self,in_shape):
        super().__init__()
        in_batch,in_length = in_shape
        #定义张量
        input = MemoryManager().allocActivation(shape=(in_batch,in_length))
        input_grad = MemoryManager().allocFeatureGrad(shape=(in_batch,in_length))
        label = MemoryManager().allocActivation(shape=(in_batch,10))
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
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr):
        return [1]
    
    def sim_run(self):
        input = self.tensors.get("input")
        loss = self.tensors.get("loss")

        input = Memory().get(input.addr)
        Memory().set(loss.addr,torch.sum(input))

class BackwardEntropy(Operator):
    def __init__(self,attrs:BackwardEntropyAttrs,tensors:BackwardEntropyTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

    def sim_run(self):
        loss = self.tensors.get("loss")
        input_grad = self.tensors.get("input_grad")

        loss = Memory().get(loss.addr)
        Memory().set(input_grad.addr,torch.ones(input_grad.shape)*1.00)