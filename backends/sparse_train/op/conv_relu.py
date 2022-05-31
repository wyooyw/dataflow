from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class ForwardConvRelu(Operator):
    def __init__(self,conv,relu):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.relu = relu

        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("output",self.relu.tensors.get("output"))
        
        self.tensors.add_read_tensor("input")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_write_tensor("relu.mask")
        self.tensors.add_write_tensor("output")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu = find_ops
        
        return ForwardConvRelu(conv=conv,relu=relu)