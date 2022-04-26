from compiler.target_gen.memory.tensor import Tensor
# from compiler.generate.operator import OperatorType
from enum import Enum
class MemOperator:
    def __init__(self,type,attr,tensors):
        self.type = type
        self.attr = attr
        self.param = param
    
    def export(self):
        return [self.type].extend(self.attr).extend(self.param)

# if __name__=="__main__":
#     print(OperatorType.SUM)
#     print(type(OperatorType.SUM)==Enum)