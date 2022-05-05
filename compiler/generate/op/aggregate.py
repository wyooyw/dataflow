from compiler.generate.dual_generator import DualGenerator
from compiler.generate.operator import Operator,OperatorType
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class AggregateDualGenerator(DualGenerator):
    def __init__(self,in_batch,
                        in_width,
                        in_height,
                        in_channels):
        #定义op
        self.forward_op = ForwardAggregate()
        self.backward_op = BackwardAggregate()

        #定义张量
        input_1 = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input_2 = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output_grad =  MemoryManager().allocGrad(shape=(in_batch,in_channels,in_height,in_width))

        self.forward.input_1.set(input_1)
        self.forward.input_2.set(input_2)
        self.backward.output_grad.set(output_grad)

class ForwardAggregate(Operator):
    def __init__(self):
        super().__init__(type=OperatorType.AGGREGATE,name=unique_class_name(self))
        self.input_1 = Pointer() #这个是此算子的输入
        self.input_2 = Pointer() #这个是此算子的输入
        self.output = Pointer()

        self._input = [self.input_1,self.input_2]
        self._output = [self.output]

class BackwardAggregate(Operator):
    def __init__(self):
        super().__init__(type=OperatorType.SPLIT,name=unique_class_name(self))
        self.output_grad = Pointer() #这个是此算子的输入
        self.input_grad_1 = Pointer()
        self.input_grad_2 = Pointer()

        self._input = [self.output_grad]
        self._output = [self.input_grad_1,self.input_grad_2]