from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import *
from compiler.generate.op.tensors.edge_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualEdge(Dual):
    def __init__(self,in_shape):
        super().__init__()
        in_batch,in_channels,in_width,in_height = in_shape

        output = MemoryManager().allocActivation(shape=in_shape)

        output_grad = MemoryManager().allocFeatureGrad(shape=in_shape)

        forward_edge_tensors = ForwardEdgeTensors(output=output)
        backward_edge_tensors = BackwardEdgeTensors(output_grad=output_grad)
    
        #定义op
        self.forward = ForwardEdge(attrs=Attrs(),
                                        tensors=forward_edge_tensors,
                                        in_shape=[],
                                        out_shape=in_shape)
        self.backward = BackwardEdge(attrs=Attrs(),
                                        tensors=backward_edge_tensors,
                                        in_shape=in_shape,
                                        out_shape=[])


class ForwardEdge(Operator):
    def __init__(self,attrs:Attrs,tensors:OpTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

class BackwardEdge(Operator):
    def __init__(self,attrs:Attrs,tensors:OpTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

