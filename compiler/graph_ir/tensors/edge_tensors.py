from compiler.graph_ir import Tensors
class ForwardEdgeTensors(Tensors):
    def __init__(self,output):
        super().__init__()
        self.tensors["output"] = output
        self.input = None
        self.output = output

class BackwardEdgeTensors(Tensors):
    def __init__(self,output_grad):
        super().__init__()
        self.tensors["output_grad"] = output_grad
        self.input = output_grad
        self.output = None