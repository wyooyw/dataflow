from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardMaxpoolTensors(OpTensors):
    def __init__(self,input,mask,output):
        super().__init__()
        self.tensors["mask"] = mask
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = input
        self.output = output

        self.add_write_tensor("mask")
        self.add_read_tensor("input")
        self.add_write_tensor("output")

class BackwardMaxpoolTensors(OpTensors):
    def __init__(self,output_grad,mask,input_grad):
        super().__init__()
        self.tensors["output_grad"] = output_grad
        self.tensors["mask"] = mask
        self.tensors["input_grad"] = input_grad
        self.input = output_grad
        self.output = input_grad

        self.add_read_tensor("output_grad")
        self.add_read_tensor("mask")
        self.add_write_tensor("input_grad")
