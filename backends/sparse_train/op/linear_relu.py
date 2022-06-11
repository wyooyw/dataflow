from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
from queue import Queue
from functools import reduce
class ForwardLinearRelu(Operator):
    def __init__(self,linear,relu):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.linear = linear
        self.relu = relu

        self.tensors.set("input",self.linear.tensors.get("input"))
        self.tensors.set("linear.weight",self.linear.tensors.get("weight"))
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("output",self.relu.tensors.get("output"))

        self.tensors.add_read_tensor("input")
        self.tensors.add_read_tensor("linear.weight")
        self.tensors.add_write_tensor("relu.mask")
        self.tensors.add_write_tensor("output")
    

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        linear,relu = find_ops
        return ForwardLinearRelu(linear=linear,relu=relu)