from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardBatchnormTensors(OpTensors):
    def __init__(self,avg,std,input,output):
        super().__init__()
        self.tensors["avg"] = avg
        self.tensors["std"] = std
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = input
        self.output = output

class BackwardBatchnormTensors(OpTensors):
    def __init__(self,avg,std,input_grad,output_grad):
        super().__init__()
        self.tensors["avg"] = avg
        self.tensors["std"] = std
        self.tensors["input_grad"] = input_grad
        self.tensors["output_grad"] = output_grad
        self.input = output_grad
        self.output = input_grad