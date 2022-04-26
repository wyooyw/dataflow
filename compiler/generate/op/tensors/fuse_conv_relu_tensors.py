from compiler.generate.op.tensors.op_tensors import OpTensors
class ForwardFuseConvReluTensors(OpTensors):
    def __init__(self,conv_weight,
                        conv_input,
                        conv_output,
                        relu_mask,
                        relu_input,
                        relu_output
                        ):
        super().__init__()
        #conv
        self.tensors["conv_weight"] = conv_weight
        self.tensors["conv_input"] = conv_input
        self.tensors["conv_output"] = conv_output
        #relu
        self.tensors["relu_mask"] = relu_mask
        self.tensors["relu_input"] = relu_input
        self.tensors["relu_output"] = relu_output


        self.input = ["conv_input"]
        self.output = ["relu_output"]

class BackwardFuseConvReluTensors(OpTensors):
    def __init__(self,conv_weight,
                        conv_input,
                        conv_output_grad,
                        conv_weight_grad,
                        conv_input_grad,
                        relu_mask,
                        relu_input_grad,
                        relu_output_grad):
        super().__init__()
        #conv
        self.tensors["conv_weight"] = conv_weight
        self.tensors["conv_input"] = conv_input
        self.tensors["conv_output_grad"] = conv_output_grad
        self.tensors["conv_weight_grad"] = conv_weight_grad
        self.tensors["conv_input_grad"] = conv_input_grad
        #relu
        self.tensors["relu_mask"] = relu_mask
        self.tensors["relu_input_grad"] = relu_input_grad
        self.tensors["relu_output_grad"] = relu_output_grad


        self.input = ["relu_output_grad"]
        self.output = ["conv_input_grad"]
