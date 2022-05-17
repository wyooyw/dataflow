from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardLinearTensors(OpTensors):
    def __init__(self,weight,input,output):
        super().__init__()
        self.tensors["weight"] = weight
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = input
        self.output = output

class BackwardLinearTensors(OpTensors):
    def __init__(self,weight,input,output_grad,weight_grad,input_grad):
        super().__init__()
        self.tensors["weight"] = weight
        self.tensors["input"] = input
        self.tensors["output_grad"] = output_grad
        self.tensors["weight_grad"] = weight_grad
        self.tensors["input_grad"] = input_grad
        self.input = output_grad
        self.output = input_grad