from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
class BackwardConvSplitReluBnBn(Operator):
    def __init__(self,conv,split,relu,bn1,bn2):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.split = split
        self.relu = relu
        self.bn1 = bn1
        self.bn2 = bn2

        self.tensors.set("output_grad",self.conv.tensors.get("output_grad"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))

        add_1,add_2 = self.split.tensors.get("output_grad1"),self.split.tensors.get("output_grad2")
        add_tensor = add_2 if self.conv.tensors.get("input_grad").storage==add_1.storage else add_1
        self.tensors.set("add",add_tensor)
        self.tensors.set("relu.mask",self.relu.tensors.get("mask"))

        self.tensors.set("bn1.bn_use",self.bn1.tensors.get("bn_use"))
        self.tensors.set("input_grad1",self.bn1.tensors.get("input_grad"))

        self.tensors.set("bn2.bn_use",self.bn2.tensors.get("bn_use"))
        self.tensors.set("input_grad2",self.bn2.tensors.get("input_grad"))

        self.tensors.add_read_tensor("output_grad")
        self.tensors.add_read_tensor("conv.weight")
        self.tensors.add_read_tensor("add")
        self.tensors.add_read_tensor("relu.mask")
        self.tensors.add_read_tensor("bn1.bn_use")
        self.tensors.add_read_tensor("bn2.bn_use")
        self.tensors.add_write_tensor("input_grad1")
        self.tensors.add_write_tensor("input_grad2")
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,split,relu,bn1,bn2 = find_ops
        return BackwardConvSplitReluBnBn(conv=conv,split=split,relu=relu,bn1=bn1,bn2=bn2)
    #这里改一下 
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
        instruction.set("output_grad",int_to_bits(self.tensors.get("output_grad").index,9).to01(),use_bits=True)
        instruction.set("weight",int_to_bits(self.tensors.get("conv.weight").index,9).to01(),use_bits=True)
        instruction.set("relu_mask",int_to_bits(self.tensors.get("relu.mask").index,9).to01(),use_bits=True)
        instruction.set("input_grad",int_to_bits(self.tensors.get("input_grad1").index,9).to01(),use_bits=True)
        return instruction