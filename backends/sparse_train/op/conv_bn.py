from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
from backends.sparse_train.target_code.instruction import Instruction
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class ForwardConvBn(Operator):
    def __init__(self,conv,bn):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.bn = bn
        
        self.tensors.set("input",self.conv.tensors.get("input"))
        self.tensors.set("conv.weight",self.conv.tensors.get("weight"))
        self.tensors.set("bn.avg",self.bn.tensors.get("avg"))
        self.tensors.set("bn.std",self.bn.tensors.get("std"))
        self.tensors.set("output",self.bn.tensors.get("output"))

    def to_instr(self):
        instruction = Instruction(init_data={
            "net_type":"resnet",
            "stage":"forward",
            "op_type":"conv",
            "stride":self.conv.attrs.get("stride"),
            "padding":False,
            "relu": False,
            "maxpool": False,
            "kernel_size":self.conv.attrs.get("kernel_size"),,
            "add":False,
            "bn":True,
            "part_sum":False,
            "softmax":False
        })
        instruction.set("in_feature",self.tensors.get("input").get_index())
        instruction.set("weight",self.tensors.get("conv.weight").get_index())
        instruction.set("bn_use",self.tensors.get("bn_use").get_index())
        instruction.set("output",self.tensors.get("output").get_index())
    # def __copy__(self):
    #     copy_conv = copy.copy(self.conv)
    #     copy_relu = copy.copy(self.relu)
    #     copy_add = copy.copy(self.add)
    #     copy_conv_relu_add = ForwardConvReluAdd(conv=copy_conv,relu=copy_relu,add=add)
    #     return copy_conv_relu_add
        
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,bn = find_ops
        return ForwardConvBn(conv=conv,bn=bn)