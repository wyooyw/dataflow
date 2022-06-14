from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
from queue import Queue
from functools import reduce
class BackwardSTConv(Operator):
    def __init__(self,conv):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.conv = conv

        self.tensors.set("output_grad",self.conv.tensors.get("output_grad"))
        self.tensors.set("weight",self.conv.tensors.get("weight"))
        self.tensors.set("input_grad",self.conv.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("weight")
        self.tensors.add_write_tensor("input_grad")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv = find_ops[0]
        return BackwardSTConv(conv=conv)
    
    def to_instr(self):
        instruction = Instruction(name=self.name,init_data={
            "net_type":"resnet",
            "stage":"backward",
            "op_type":"conv",
            "stride":self.conv.attrs.get("stride"),
            "padding":False,
            "relu": True,
            "maxpool": False,
            "kernel_size":self.conv.attrs.get("kernel_size"),
            "add":False,
            "bn":False,
            "part_sum":False,
            "softmax":False
        },pad_to=128)
        instruction.set("in_feature",int_to_bits(self.tensors.get("output_grad").index,9).to01(),use_bits=True)
        instruction.set("weight",int_to_bits(self.tensors.get("weight").index,9).to01(),use_bits=True)
        instruction.set("output",int_to_bits(self.tensors.get("input_grad").index,9).to01(),use_bits=True)
        return instruction