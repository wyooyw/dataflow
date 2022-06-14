from compiler.graph_ir import Operator,OperatorType,Attrs,Tensors
from compiler.utils.unique_class_name import unique_class_name
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
from queue import Queue
from functools import reduce
class ForwardLinearSoftmaxEntropy(Operator):
    def __init__(self,linear,softmax,entropy,back_entropy,back_softmax):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=Attrs(),
                        tensors=Tensors(),
                        name=unique_class_name(self))
        self.linear = linear
        self.softmax = softmax
        self.entropy = entropy
        self.back_entropy = back_entropy
        self.back_softmax = back_softmax
    

        self.tensors.set("input",self.linear.tensors.get("input"))
        self.tensors.set("linear.weight",self.linear.tensors.get("weight"))
        self.tensors.set("output",self.back_softmax.tensors.get("input_grad"))

        self.tensors.add_read_tensor("input")
        self.tensors.add_read_tensor("linear.weight")
        self.tensors.add_write_tensor("output")
    
    def to_instr(self):
        instruction = Instruction(name=self.name, init_data={
            "net_type": "resnet",
            "stage": "forward",
            "op_type": "linear",
            "stride": 1,
            "padding": False,
            "relu": False,
            "maxpool": False,
            "kernel_size": 1,
            "add": False,
            "bn": False,
            "part_sum": False,
            "softmax": True
        },pad_to=128)
        instruction.set("in_feature", int_to_bits(self.tensors.get("input").index,9).to01(),use_bits=True)
        instruction.set("weight", int_to_bits(self.tensors.get("linear.weight").index,9).to01(),use_bits=True)
        instruction.set("output", int_to_bits(self.tensors.get("output").index,9).to01(),use_bits=True)
        return instruction
        
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        linear,softmax,entropy,back_entropy,back_softmax = find_ops
        return ForwardLinearSoftmaxEntropy(linear=linear,softmax=softmax,entropy=entropy,back_entropy=back_entropy,back_softmax=back_softmax)