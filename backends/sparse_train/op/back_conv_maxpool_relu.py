from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardConvMaxpoolRelu(Operator):
    def __init__(self,conv,maxpool,relu):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.maxpool = maxpool
        self.relu = relu

        self.tensors.set("output_grad",conv.tensors.get("output_grad"))
        self.tensors.set("conv.weight",conv.tensors.get("weight"))

        self.tensors.set("maxpool.mask",maxpool.tensors.get("mask"))
        self.tensors.set("relu.mask",relu.tensors.get("mask"))

        self.tensors.set("input_grad",relu.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_read_tensor("relu.mask")
        self.tensors.add_read_tensor("maxpool.mask")
        self.tensors.add_write_tensor("input_grad")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,maxpool,relu = find_ops
        return BackwardConvMaxpoolRelu(conv=conv,maxpool=maxpool,relu=relu)