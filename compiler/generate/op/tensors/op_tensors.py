import collections
import copy
class OpTensors:
    """算子中使用到的
    """
    def __init__(self):
        self.tensors = collections.OrderedDict()
        self.input = None
        self.output = None
        self.output_idx = -1
        self.input_idx = -1
        self.op = None #该OpTensors的父元素

    def get_input(self):
        return [self.tensors[item] for item in self.input]

    def get_output(self):
        return [self.tensors[item] for item in self.output]

    def set_next_output(self,tensor):
        self.output_idx += 1
        assert self.output_idx < len(self.output),f"[{self.op.name}] output out of range,output_idx={self.output_idx},len(self.output)={len(self.output)}"
        self.tensors[self.output[self.output_idx]] = tensor

    def get_next_input(self):
        self.input_idx += 1
        assert self.input_idx < len(self.input),f"[{self.op.name}] input out of range,input_idx={self.input_idx},len(self.input)={len(self.input)}"
        return self.tensors[self.input[self.input_idx]]
    
    def get(self,key):
        return self.tensors[key]
    
    def set(self,key,value):
        self.tensors[key] = value

    def __copy__(self):
        """复制对象

        这里复制后,type(copy_self)会变成OpTensor,而不是子类,目前不影响,后续最好处理一下。
        """
        copy_self = OpTensors()
        copy_self.tensors = copy.copy(self.tensors)
        copy_self.input = copy.copy(self.input)
        copy_self.output = copy.copy(self.output)
        copy_self.output_idx = self.output_idx
        copy_self.input_idx = self.input_idx
        copy_self.op = self.op
        return copy_self
        
