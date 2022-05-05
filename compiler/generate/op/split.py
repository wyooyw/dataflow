from compiler.generate.dual_generator import DualGenerator
from compiler.generate.operator import Operator,OperatorType
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class SplitDualGenerator(DualGenerator):
    def __init__(self,in_batch,
                        in_width,
                        in_height,
                        in_channels):
        #定义op
        self.forward_op = ForwardSplit()
        self.backward_op = BackwardSplit()

        #定义张量
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output_grad_1 =  MemoryManager().allocGrad(shape=(in_batch,in_channels,in_height,in_width))
        output_grad_2 =  MemoryManager().allocGrad(shape=(in_batch,in_channels,in_height,in_width))

        self.forward.input.set(input)  
        self.backward.output_grad_1.set(output_grad_1)   
        self.backward.output_grad_2.set(output_grad_2)

class ForwardSplit(Operator):
    def __init__(self):
        super().__init__(type=OperatorType.SPLIT,name=unique_class_name(self))
        self.input = Pointer() #这个是此算子的输入
        self.output_1 = Pointer()
        self.output_2 = Pointer()

        self._input = [self.input]
        self._output = [self.output_1,self.output_2]

class BackwardSplit(Operator):
    def __init__(self):
        super().__init__(type=OperatorType.AGGREGATE,name=unique_class_name(self))
        self.output_grad_1 = Pointer() #这个是此算子的输入
        self.output_grad_2 = Pointer() #这个是此算子的输入
        self.input_grad = Pointer()

        self._input = [self.output_grad_1,self.output_grad_2]
        self._output = [self.input_grad]