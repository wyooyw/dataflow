from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardReluTensors(OpTensors):
    def __init__(self,mask,input,output):
        super().__init__()
        self.tensors["mask"] = mask
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = input
        self.output = output
        
        self.add_read_tensor("input")
        self.add_write_tensor("mask")
        self.add_write_tensor("output")

class BackwardReluTensors(OpTensors):
    def __init__(self,mask,input_grad,output_grad):
        super().__init__()
        self.tensors["mask"] = mask
        self.tensors["input_grad"] = input_grad
        self.tensors["output_grad"] = output_grad
        self.input = output_grad
        self.output = input_grad

        self.add_read_tensor("output_grad")
        self.add_read_tensor("mask")
        self.add_write_tensor("input_grad")