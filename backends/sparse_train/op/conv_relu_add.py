from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.conv import ForwardConv
from compiler.generate.op.relu import ForwardRelu
from compiler.generate.op.add import ForwardAdd
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from queue import Queue
from functools import reduce
class ForwardConvReluAdd(Operator):
    def __init__(self,conv,relu,add):
        super().__init__(type=OperatorType.BACKEND,
                        attrs=Attrs(),
                        tensors=OpTensors(),
                        name=unique_class_name(self))
        self.conv = conv
        self.relu = relu
        self.add = add

    def __copy__(self):
        copy_conv = copy.copy(self.conv)
        copy_relu = copy.copy(self.relu)
        copy_add = copy.copy(self.add)
        copy_conv_relu_add = ForwardConvReluAdd(conv=copy_conv,relu=copy_relu,add=add)
        return copy_conv_relu_add
        
    @classmethod
    def replace_from(self,find_ops):
        """将ForwardConv和ForwardRelu合并为ForwardConvRelu
        """
        conv,relu,add = find_ops
        return ForwardConvReluAdd(conv=conv,relu=relu,add=add)