from compiler.graph_ir import Tensors
class ForwardAddTensors(Tensors):
    def __init__(self,input1,input2,output):
        super().__init__()
        self.tensors["input1"] = input1
        self.tensors["input2"] = input2
        self.tensors["output"] = output
        self.add_read_tensor("input1")
        self.add_read_tensor("input2")
        self.add_write_tensor("output")
        self.input = [input1,input2]
        self.output = output

class BackwardAddTensors(Tensors):
    def __init__(self,grad):
        super().__init__()
        self.tensors["grad"] = grad
        self.input = grad
        self.output = grad