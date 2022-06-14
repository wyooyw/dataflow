from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardLinearMaxpoolReluAddBn(Operator):
    def __init__(self,linear,maxpool,relu,add,bn):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.linear = linear
        self.maxpool = maxpool
        self.relu = relu
        self.add = add
        self.bn = bn

        self.tensors.set("output_grad",self.linear.tensors.get("output_grad"))
        self.tensors.set("conv.weight",self.linear.tensors.get("weight"))

        add_1,add_2 = self.split.tensors.get("output_grad1"),self.split.tensors.get("output_grad2")
        add_tensor = add_2 if self.conv.tensors.get("input_grad").storage==add_1.storage else add_1
        self.tensors.set("add",add_tensor)
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("maxpool.mask",self.maxpool.tensors.get("mask"))

        self.tensors.set("bn.bn_use",self.bn.tensors.get("bn_use"))
        self.tensors.set("input_grad",self.bn.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_read_tensor("add")
        self.tensors.add_read_tensor("relu.mask")
        self.tensors.add_read_tensor("bn.bn_use")
        self.tensors.add_write_tensor("input_grad")

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,split,relu,bn = find_ops
        return BackwardConvSplitReluBn(conv=conv,split=split,relu=relu,bn=bn)