from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class BackwardSTConv(Operator):
    def __init__(self,conv):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv

        self.tensors.set("output_grad",self.conv.tensors.get("output_grad"))
        self.tensors.set("weight",self.conv.tensors.get("weight"))
        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("input_grad",self.conv.tensors.get("input_grad"))
        self.tensors.set("weight_grad",self.conv.tensors.get("weight_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("weight")
        self.tensors.add_read_tensor("input")
        self.tensors.add_write_tensor("weight_grad")
        self.tensors.add_write_tensor("input_grad")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv = find_ops[0]
        return BackwardSTConv(conv=conv)