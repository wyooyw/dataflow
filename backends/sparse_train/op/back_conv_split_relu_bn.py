from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardConvSplitReluBn(Operator):
    def __init__(self,conv,split,relu,bn):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.split = split
        self.relu = relu
        self.bn = bn

        self.tensors.set("output_grad",self.conv.tensors.get("output_grad"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("conv.input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight_grad",self.conv.tensors.get("weight_grad"))

        add_1,add_2 = self.split.tensors.get("output_grad1"),self.split.tensors.get("output_grad2")
        add_tensor = add_2 if self.conv.tensors.get("input_grad").storage==add_1.storage else add_1
        self.tensors.set("add",add_tensor)
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))

        self.tensors.set("bn.avg",self.bn.tensors.get("avg"))
        self.tensors.set("bn.std",self.bn.tensors.get("std"))
        self.tensors.set("input_grad",self.bn.tensors.get("input_grad"))

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,split,relu,bn = find_ops
        return BackwardConvSplitReluBn(conv=conv,split=split,relu=relu,bn=bn)