from compiler.graph_ir import Tensors
class ForwardSoftmaxTensors(Tensors):
    def __init__(self,input,output):
        super().__init__()
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = input
        self.output = output

        self.add_read_tensor("input")
        self.add_write_tensor("output")

class BackwardSoftmaxTensors(Tensors):
    def __init__(self,input_grad,output_grad):
        super().__init__()
        self.tensors["input_grad"] = input_grad
        self.tensors["output_grad"] = output_grad
        self.input = output_grad
        self.output = input_grad

        self.add_read_tensor("output_grad")
        self.add_write_tensor("input_grad")