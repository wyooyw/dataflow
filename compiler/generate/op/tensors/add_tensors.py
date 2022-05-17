from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardAddTensors(OpTensors):
    def __init__(self,input1,input2,output):
        super().__init__()
        self.tensors["input1"] = input1
        self.tensors["input2"] = input2
        self.tensors["output"] = output
        self.input = [input1,input2]
        self.output = output

class BackwardAddTensors(OpTensors):
    def __init__(self,grad):
        super().__init__()
        self.tensors["grad"] = grad
        self.input = grad
        self.output = grad