from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardConvSplitReluBnBn(Operator):
    def __init__(self,conv,split,relu,bn1,bn2):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.split = split
        self.relu = relu
        self.bn1 = bn1
        self.bn2 = bn2

        self.tensors.set("output_grad",self.conv.tensors.get("output_grad"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("conv.input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight_grad",self.conv.tensors.get("weight_grad"))

        add_1,add_2 = self.split.tensors.get("output_grad1"),self.split.tensors.get("output_grad2")
        add_tensor = add_2 if self.conv.tensors.get("input_grad").storage==add_1.storage else add_1
        self.tensors.set("add",add_tensor)
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))

        self.tensors.set("bn1.bn_use",self.bn1.tensors.get("bn_use"))
        self.tensors.set("input_grad1",self.bn1.tensors.get("input_grad"))

        self.tensors.set("bn2.bn_use",self.bn2.tensors.get("bn_use"))
        self.tensors.set("input_grad2",self.bn2.tensors.get("input_grad"))

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,split,relu,bn1,bn2 = find_ops
        return BackwardConvSplitReluBnBn(conv=conv,split=split,relu=relu,bn1=bn1,bn2=bn2)