from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardSplitTensors(OpTensors):
    def __init__(self,input):
        super().__init__()
        self.tensors["input"] = input
        self.input = input
        self.output = input

class BackwardSplitTensors(OpTensors):
    def __init__(self,output_grad1,output_grad2,input_grad):
        super().__init__()
        self.tensors["output_grad1"] = output_grad1
        self.tensors["output_grad2"] = output_grad2
        self.tensors["input_grad"] = input_grad
        self.input = [output_grad1,output_grad2]
        self.output = input_grad

        self.add_read_tensor("output_grad1")
        self.add_read_tensor("output_grad2")
        self.add_write_tensor("input_grad")