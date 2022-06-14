from compiler.graph_ir import Tensors
class ForwardBatchnormTensors(Tensors):
    def __init__(self,avg,std,alpha,beta,input,output):
        super().__init__()
        self.tensors["avg"] = avg
        self.tensors["std"] = std
        self.tensors["alpha"] = alpha
        self.tensors["beta"] = beta
        self.tensors["input"] = input
        self.tensors["output"] = output

        self.add_read_tensor("avg")
        self.add_read_tensor("std")
        self.add_read_tensor("alpha")
        self.add_read_tensor("beta")
        self.add_read_tensor("input")
        self.add_write_tensor("output")

        self.input = input
        self.output = output

class BackwardBatchnormTensors(Tensors):
    def __init__(self,avg,std,alpha,beta,input_grad,output_grad):
        super().__init__()
        self.tensors["avg"] = avg
        self.tensors["std"] = std
        self.tensors["alpha"] = alpha
        self.tensors["beta"] = beta
        self.tensors["input_grad"] = input_grad
        self.tensors["output_grad"] = output_grad
        self.input = output_grad
        self.output = input_grad

        self.add_read_tensor("avg")
        self.add_read_tensor("std")
        self.add_read_tensor("alpha")
        self.add_read_tensor("beta")
        self.add_read_tensor("output_grad")
        self.add_write_tensor("input_grad")