from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class ForwardConvReluBnBn(Operator):
    def __init__(self,conv,relu,bn1,bn2):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.bn1 = bn1
        self.bn2 = bn2
        self.relu = relu

        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("bn1.bn_use",self.bn1.tensors.get("bn_use"))
        self.tensors.set("bn2.bn_use",self.bn2.tensors.get("bn_use"))
        self.tensors.set("output1",self.bn1.tensors.get("output"))
        self.tensors.set("output2",self.bn2.tensors.get("output"))
        
        self.tensors.add_read_tensor("input")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_read_tensor("bn1.bn_use")
        self.tensors.add_read_tensor("bn2.bn_use")
        self.tensors.add_write_tensor("relu.mask")
        self.tensors.add_write_tensor("output1")
        self.tensors.add_write_tensor("output2")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu,bn1,bn2 = find_ops
        
        return ForwardConvReluBnBn(conv=conv,bn1=bn1,bn2=bn2,relu=relu)