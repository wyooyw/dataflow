from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
from queue import Queue
from functools import reduce
class ForwardConvBnRelu(Operator):
    def __init__(self,conv,bn,relu):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("bn.bn_use",self.bn.tensors.get("bn_use"))
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))
        self.tensors.set("output",self.relu.tensors.get("output"))
    
        self.tensors.add_read_tensor("input")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_read_tensor("bn.bn_use")
        self.tensors.add_write_tensor("relu.mask")
        self.tensors.add_write_tensor("output")
    def to_instr(self):
        instruction = Instruction(name=self.name,init_data={
            "net_type":"resnet",
            "stage":"forward",
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
        })
        instruction.set("in_feature",int_to_bits(self.tensors.get("input").index,9).to01(),use_bits=True)
        instruction.set("weight",int_to_bits(self.tensors.get("conv.weight").index,9).to01(),use_bits=True)
        instruction.set("relu_mask",int_to_bits(self.tensors.get("relu.mask").index,9).to01(),use_bits=True)
        instruction.set("bn_use",int_to_bits(self.tensors.get("bn.bn_use").index,9).to01(),use_bits=True)
        instruction.set("output",int_to_bits(self.tensors.get("output").index,9).to01(),use_bits=True)
        return instruction

    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,bn,relu = find_ops
        
        return ForwardConvBnRelu(conv=conv,bn=bn,relu=relu)