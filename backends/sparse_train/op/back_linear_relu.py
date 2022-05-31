from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardLinearRelu(Operator):
    def __init__(self,linear,relu):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.linear = linear
        self.relu = relu

        self.tensors.set("output_grad",linear.tensors.get("output_grad"))
        self.tensors.set("linear.weight",linear.tensors.get("weight"))
        self.tensors.set("linear.input",linear.tensors.get("input"))
        self.tensors.set("linear.weight_grad",linear.tensors.get("weight_grad"))

        self.tensors.set("relu.mask",relu.tensors.get("mask"))

        self.tensors.set("input_grad",relu.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("linear.weight")
        self.tensors.add_read_tensor("linear.input")
        self.tensors.add_read_tensor("relu.mask")
        self.tensors.add_write_tensor("linear.weight_grad")
        self.tensors.add_write_tensor("input_grad")

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        linear,relu = find_ops
        return BackwardLinearRelu(linear=linear,relu=relu)