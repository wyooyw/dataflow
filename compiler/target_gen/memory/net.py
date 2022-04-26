from compiler.target_gen.memory.mem_operator import MemOperator

class Net:
    def __init__(self,operator:list[MemOperator]==[]):
        self.operator = operator
    