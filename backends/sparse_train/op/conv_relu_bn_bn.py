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
        self.tensors.set("bn1.avg",self.bn1.tensors.get("avg"))
        self.tensors.set("bn1.std",self.bn1.tensors.get("std"))
        self.tensors.set("bn2.avg",self.bn2.tensors.get("avg"))
        self.tensors.set("bn2.std",self.bn2.tensors.get("std"))
        self.tensors.set("output1",self.bn1.tensors.get("output"))
        self.tensors.set("output2",self.bn2.tensors.get("output"))
        
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu,bn1,bn2 = find_ops
        
        return ForwardConvReluBnBn(conv=conv,bn1=bn1,bn2=bn2,relu=relu)