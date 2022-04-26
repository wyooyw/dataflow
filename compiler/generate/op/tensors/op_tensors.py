import collections
class OpTensors:
    """算子中使用到的
    """
    def __init__(self):
        self.tensors = collections.OrderedDict()
        self.input = []
        self.output = []
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