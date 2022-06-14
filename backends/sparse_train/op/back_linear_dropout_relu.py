from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardLinearDropoutRelu(Operator):
    def __init__(self,linear,dropout,relu):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.linear = linear
        self.dropout = dropout
        self.relu = relu

        self.tensors.set("output_grad",linear.tensors.get("output_grad"))
        self.tensors.set("linear.weight",linear.tensors.get("weight"))

        self.tensors.set("dropout.mask",dropout.tensors.get("mask"))
        self.tensors.set("relu.mask",relu.tensors.get("mask"))

        self.tensors.set("input_grad",relu.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("linear.weight")
        self.tensors.add_read_tensor("relu.mask")
        self.tensors.add_read_tensor("dropout.mask")
        self.tensors.add_write_tensor("input_grad")
        
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        linear,dropout,relu = find_ops
        return BackwardLinearDropoutRelu(linear=linear,dropout=dropout,relu=relu)