from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardAddTensors(OpTensors):
    def __init__(self,input1,input2,output):
        super().__init__()
        self.tensors["input1"] = input1
        self.tensors["input2"] = input2
        self.tensors["output"] = output
        self.input = ["input1","input2"]
        self.output = ["output"]

class BackwardAddTensors(OpTensors):
    def __init__(self,output_grad,input_grad1,input_grad2):
        super().__init__()
        self.tensors["output_grad"] = output_grad
        self.tensors["input_grad1"] = input_grad1
        self.tensors["input_grad2"] = input_grad2
        self.input = ["output_grad"]
        self.output = ["input_grad1","input_grad2"]