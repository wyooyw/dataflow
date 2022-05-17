from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class ForwardConvBnAddRelu(Operator):
    def __init__(self,conv,bn,add,relu):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.bn = bn
        self.add = add
        self.relu = relu
        
        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("bn.avg",self.bn.tensors.get("avg"))
        self.tensors.set("bn.std",self.bn.tensors.get("std"))
        add_1,add_2 = self.add.tensors.get("input1"),self.add.tensors.get("input2")
        add_tensor = add_2 if self.bn.tensors.get("output").storage==add_1.storage else add_1
        self.tensors.set("add",add_tensor)
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("output",self.relu.tensors.get("output"))

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,bn,add,relu = find_ops
        return ForwardConvBnAddRelu(conv=conv,bn=bn,add=add,relu=relu)