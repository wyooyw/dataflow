from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardConvReluBn(Operator):
    def __init__(self,conv,relu,bn):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.relu = relu
        self.bn = bn

        self.tensors.set("output_grad",conv.tensors.get("output_grad"))
        self.tensors.set("conv.weight",conv.tensors.get("weight"))
        self.tensors.set("conv.input",conv.tensors.get("input"))
        self.tensors.set("conv.weight_grad",conv.tensors.get("weight_grad"))

        self.tensors.set("relu.mask",relu.tensors.get("mask"))

        self.tensors.set("bn.avg",bn.tensors.get("avg"))
        self.tensors.set("bn.std",bn.tensors.get("std"))
        self.tensors.set("input_grad",bn.tensors.get("input_grad"))

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu,bn = find_ops
        return BackwardConvReluBn(conv=conv,bn=bn,relu=relu)