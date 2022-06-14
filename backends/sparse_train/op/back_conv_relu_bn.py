from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
from queue import Queue
from functools import reduce
class BackwardConvReluBn(Operator):
    def __init__(self,conv,relu,bn):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.relu = relu
        self.bn = bn

        self.tensors.set("output_grad",conv.tensors.get("output_grad"))
        self.tensors.set("conv.weight",conv.tensors.get("weight"))

        self.tensors.set("relu.mask",relu.tensors.get("mask"))

        self.tensors.set("bn.bn_use",bn.tensors.get("bn_use"))
        self.tensors.set("input_grad",bn.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_read_tensor("relu.mask")
        self.tensors.add_read_tensor("bn.bn_use")
        self.tensors.add_write_tensor("input_grad")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu,bn = find_ops
        return BackwardConvReluBn(conv=conv,bn=bn,relu=relu)

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
            "bn":True,
            "part_sum":False,
            "softmax":False
        },pad_to=128)
        instruction.set("output_grad",int_to_bits(self.tensors.get("output_grad").index,9).to01(),use_bits=True)
        instruction.set("weight",int_to_bits(self.tensors.get("conv.weight").index,9).to01(),use_bits=True)
        instruction.set("relu_mask",int_to_bits(self.tensors.get("relu.mask").index,9).to01(),use_bits=True)
        instruction.set("bn_use",int_to_bits(self.tensors.get("bn.bn_use").index,9).to01(),use_bits=True)
        instruction.set("input_grad",int_to_bits(self.tensors.get("input_grad").index,9).to01(),use_bits=True)
        return instruction