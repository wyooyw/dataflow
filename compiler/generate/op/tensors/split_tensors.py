from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardSplitTensors(OpTensors):
    def __init__(self,input,output_1,output_2):
        super().__init__()
        self.tensors["input"] = input
        self.tensors["output_1"] = output_1
        self.tensors["output_2"] = output_2
        self.input = ["input"]
        self.output = ["output1","output2"]

class BackwardSplitTensors(OpTensors):
    def __init__(self,output_grad_1,output_grad_2,input_grad):
        super().__init__()
        self.tensors["output_grad_1"] = output_grad_1
        self.tensors["output_grad_2"] = output_grad_2
        self.tensors["input_grad"] = input_grad
        self.input = ["output_grad_1","output_grad_2"]
        self.output = ["input_grad"]