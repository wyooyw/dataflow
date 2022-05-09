from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardEntropyTensors(OpTensors):
    def __init__(self,label,input,loss):
        super().__init__()
        self.tensors["label"] = label
        self.tensors["input"] = input
        self.tensors["loss"] = loss
        self.input = ["input"]
        self.output = ["loss"]

class BackwardEntropyTensors(OpTensors):
    def __init__(self,label,input_grad,loss):
        super().__init__()
        self.tensors["label"] = label
        self.tensors["input_grad"] = input_grad
        self.tensors["loss"] = loss
        self.input = ["loss"]
        self.output = ["input_grad"]