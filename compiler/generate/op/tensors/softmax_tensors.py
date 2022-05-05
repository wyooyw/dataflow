from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardSoftmaxTensors(OpTensors):
    def __init__(self,input,output):
        super().__init__()
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = ["input"]
        self.output = ["output"]

class BackwardSoftmaxTensors(OpTensors):
    def __init__(self,input_grad,output_grad):
        super().__init__()
        self.tensors["input_grad"] = input_grad
        self.tensors["output_grad"] = output_grad
        self.input = ["output_grad"]
        self.output = ["input_grad"]