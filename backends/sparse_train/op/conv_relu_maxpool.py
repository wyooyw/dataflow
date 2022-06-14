from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
from queue import Queue
from functools import reduce
class ForwardConvReluMaxpool(Operator):
    def __init__(self,conv,relu,maxpool):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.relu = relu
        self.maxpool = maxpool

        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("maxpool.mask",self.maxpool.tensors.get("mask"))
        self.tensors.set("output",self.maxpool.tensors.get("output"))
    
        self.tensors.add_read_tensor("input")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_write_tensor("relu.mask")
        self.tensors.add_write_tensor("maxpool.mask")
        self.tensors.add_write_tensor("output")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu,maxpool = find_ops
        return ForwardConvReluMaxpool(conv=conv,relu=relu,maxpool=maxpool)