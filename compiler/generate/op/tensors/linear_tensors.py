from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardLinearTensors(OpTensors):
    def __init__(self,weight,input,output):
        super().__init__()
        self.tensors["weight"] = weight
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.input = input
        self.output = output

        self.add_read_tensor("weight")
        self.add_read_tensor("input")
        self.add_write_tensor("output")

class BackwardLinearTensors(OpTensors):
    def __init__(self,weight,output_grad,input_grad):
        super().__init__()
        self.tensors["weight"] = weight
        # self.tensors["input"] = input
        self.tensors["output_grad"] = output_grad
        # self.tensors["weight_grad"] = weight_grad
        self.tensors["input_grad"] = input_grad
        self.input = output_grad
        self.output = input_grad

        self.add_read_tensor("weight")
        # self.add_read_tensor("input")
        self.add_read_tensor("output_grad")
        # self.add_write_tensor("weight_grad")
        self.add_write_tensor("input_grad")


"""权重梯度算子用到的张量
"""
class WGLinearTensors(OpTensors):
    def __init__(self,input,output_grad,weight_grad):
        super().__init__()
        self.tensors["input"] = input
        self.tensors["output_grad"] = output_grad
        self.tensors["weight_grad"] = weight_grad
        self.add_read_tensor("input")
        self.add_read_tensor("output_grad")
        self.add_write_tensor("weight_grad")
        self.input = output_grad
        self.output = None
    
"""权重更新算子用到的张量
"""
class WULinearTensors(OpTensors):
    def __init__(self,weight,weight_grad):
        super().__init__()
        self.tensors["weight"] = weight
        self.tensors["weight_grad"] = weight_grad
        self.add_read_tensor("weight")
        self.add_write_tensor("weight")
        self.add_write_tensor("weight_grad")
