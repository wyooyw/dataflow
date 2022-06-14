

"""
conv相关的前传、反传算子，用到的属性、张量，以及产生一对算子的工厂类
"""

class ConvDualGenerator(DualGenerator):
    """ 同时产生Conv的前传和反传算子
    """
    def __init__(self,in_batch,
                        in_width,
                        in_height,
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=0,
                        ):
                        
        # 定义张量
        weight = MemoryManager().allocWeight(shape=(out_channels,in_channels,kernel_size,kernel_size))
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output = MemoryManager().allocGrad(shape=(in_batch,out_channels,in_width+2*padding-kernel_size+1,in_width+2*padding-kernel_size+1))
        weight_grad = MemoryManager().allocWeight(shape=(out_channels,in_channels,kernel_size,kernel_size))
        input_grad = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output_grad = MemoryManager().allocGrad(shape=(in_batch,out_channels,in_width+2*padding-kernel_size+1,in_width+2*padding-kernel_size+1))
        # 前传参数
        forward_conv_attrs = ForwardConvAttrs(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        )
        # 反传参数
        backward_conv_attrs = BackwardConvAttrs(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        )

        # 前传张量
        forward_conv_tensors = ForwardConvTensors(weight=weight,
                                                        input=input,
                                                        output=output,
                                                        )
        backward_conv_tensors = BackwardConvTensors(weight=weight,
                                                        input=input,
                                                        output_grad=output_grad,
                                                        weight_grad=weight_grad,
                                                        input_grad=input_grad,
                                                        )

        #定义op
        self.forward_op = ForwardConv(attrs=forward_conv_attrs,
                                        tensors=forward_conv_tensors)
        self.backward_op = BackwardConv(attrs=backward_conv_attrs,
                                        tensors=backward_conv_tensors)

class ForwardConv(Operator):
    """前传Conv算子
    """
    def __init__(self,attrs:ForwardConvAttrs,tensors:ForwardConvTensors):
        super().__init__(type=OperatorType.FORWARD_,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class BackwardConv(Operator):
    """反传Conv算子
    """
    def __init__(self,attrs:BackwardConvAttrs,tensors:BackwardConvTensors):
        super().__init__(type=OperatorType.BACKWARD_,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class ForwardConvAttrs(Attrs):
    """前传Conv算子的属性
    """
    def __init__(self,in_channels,
                        out_channels,
                        kernel_size,
                        padding,
                        ):
        super().__init__()
        self.attrs["in_channels"] = in_channels
        self.attrs["out_channels"] = out_channels
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding
        

class BackwardConvAttrs(Attrs):
    """反传Conv算子的属性
    """
    def __init__(self,in_channels,
                        out_channels,
                        kernel_size,
                        padding,
                        ):
        super().__init__()
        self.attrs["in_channels"] = in_channels
        self.attrs["out_channels"] = out_channels
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding
        

class ForwardConvTensors(Tensors):
    """前传Conv算子 用到的张量
    """
    def __init__(self,weight,
                        input,
                        output,
                        ):
        super().__init__()
        self.tensors["weight"] = weight
        self.tensors["input"] = input
        self.tensors["output"] = output
        
        self.input = ["input",]
        self.output = ["output",]

class BackwardConvTensors(Tensors):
    """反传Conv算子 用到的张量
    """
    def __init__(self,weight,
                        input,
                        output_grad,
                        weight_grad,
                        input_grad,
                        ):
        super().__init__()
        self.tensors["weight"] = weight
        self.tensors["input"] = input
        self.tensors["output_grad"] = output_grad
        self.tensors["weight_grad"] = weight_grad
        self.tensors["input_grad"] = input_grad
        
        self.input = ["output_grad",]
        self.output = ["input_grad",]



"""
relu相关的前传、反传算子，用到的属性、张量，以及产生一对算子的工厂类
"""

class ReluDualGenerator(DualGenerator):
    """ 同时产生Relu的前传和反传算子
    """
    def __init__(self,in_batch,
                        in_width,
                        in_height,
                        in_channels,
                        ):
                        
        # 定义张量
        mask = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input_grad = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        output_grad = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        # 前传参数
        forward_relu_attrs = ForwardReluAttrs()
        # 反传参数
        backward_relu_attrs = BackwardReluAttrs()

        # 前传张量
        forward_relu_tensors = ForwardReluTensors(mask=mask,
                                                        input=input,
                                                        output=output,
                                                        )
        backward_relu_tensors = BackwardReluTensors(mask=mask,
                                                        output_grad=output_grad,
                                                        input_grad=input_grad,
                                                        )

        #定义op
        self.forward_op = ForwardRelu(attrs=forward_relu_attrs,
                                        tensors=forward_relu_tensors)
        self.backward_op = BackwardRelu(attrs=backward_relu_attrs,
                                        tensors=backward_relu_tensors)

class ForwardRelu(Operator):
    """前传Relu算子
    """
    def __init__(self,attrs:ForwardReluAttrs,tensors:ForwardReluTensors):
        super().__init__(type=OperatorType.FORWARD_,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class BackwardRelu(Operator):
    """反传Relu算子
    """
    def __init__(self,attrs:BackwardReluAttrs,tensors:BackwardReluTensors):
        super().__init__(type=OperatorType.BACKWARD_,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class ForwardReluAttrs(Attrs):
    """前传Relu算子的属性
    """
    def __init__(self,):
        super().__init__()
        

class BackwardReluAttrs(Attrs):
    """反传Relu算子的属性
    """
    def __init__(self,):
        super().__init__()
        

class ForwardReluTensors(Tensors):
    """前传Relu算子 用到的张量
    """
    def __init__(self,mask,
                        input,
                        output,
                        ):
        super().__init__()
        self.tensors["mask"] = mask
        self.tensors["input"] = input
        self.tensors["output"] = output
        
        self.input = ["input",]
        self.output = ["output",]

class BackwardReluTensors(Tensors):
    """反传Relu算子 用到的张量
    """
    def __init__(self,mask,
                        output_grad,
                        input_grad,
                        ):
        super().__init__()
        self.tensors["mask"] = mask
        self.tensors["output_grad"] = output_grad
        self.tensors["input_grad"] = input_grad
        
        self.input = ["output_grad",]
        self.output = ["input_grad",]



"""
linear相关的前传、反传算子，用到的属性、张量，以及产生一对算子的工厂类
"""

class LinearDualGenerator(DualGenerator):
    """ 同时产生Linear的前传和反传算子
    """
    def __init__(self,in_batch,
                        in_features,
                        out_features,
                        ):
                        
        # 定义张量
        input = MemoryManager().allocActivation(shape=(in_batch,in_features))
        output = MemoryManager().allocActivation(shape=(in_batch,out_features))
        weight = MemoryManager().allocWeight(shape=(in_features,out_features))
        input_grad = MemoryManager().allocActivation(shape=(in_batch,in_features))
        output_grad = MemoryManager().allocActivation(shape=(in_batch,out_features))
        weight_grad = MemoryManager().allocWeight(shape=(in_features,out_features))
        # 前传参数
        forward_linear_attrs = ForwardLinearAttrs(in_batch=in_batch,
                                                        in_features=in_features,
                                                        out_features=out_features,
                                                        )
        # 反传参数
        backward_linear_attrs = BackwardLinearAttrs(in_batch=in_batch,
                                                        in_features=in_features,
                                                        out_features=out_features,
                                                        )

        # 前传张量
        forward_linear_tensors = ForwardLinearTensors(input=input,
                                                        output=output,
                                                        weight=weight,
                                                        )
        backward_linear_tensors = BackwardLinearTensors(input_grad=input_grad,
                                                        output_grad=output_grad,
                                                        weight_grad=weight_grad,
                                                        input=input,
                                                        weight=weight,
                                                        )

        #定义op
        self.forward_op = ForwardLinear(attrs=forward_linear_attrs,
                                        tensors=forward_linear_tensors)
        self.backward_op = BackwardLinear(attrs=backward_linear_attrs,
                                        tensors=backward_linear_tensors)

class ForwardLinear(Operator):
    """前传Linear算子
    """
    def __init__(self,attrs:ForwardLinearAttrs,tensors:ForwardLinearTensors):
        super().__init__(type=OperatorType.FORWARD_,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class BackwardLinear(Operator):
    """反传Linear算子
    """
    def __init__(self,attrs:BackwardLinearAttrs,tensors:BackwardLinearTensors):
        super().__init__(type=OperatorType.BACKWARD_,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

class ForwardLinearAttrs(Attrs):
    """前传Linear算子的属性
    """
    def __init__(self,in_batch,
                        in_features,
                        out_features,
                        ):
        super().__init__()
        self.attrs["in_batch"] = in_batch
        self.attrs["in_features"] = in_features
        self.attrs["out_features"] = out_features
        

class BackwardLinearAttrs(Attrs):
    """反传Linear算子的属性
    """
    def __init__(self,in_batch,
                        in_features,
                        out_features,
                        ):
        super().__init__()
        self.attrs["in_batch"] = in_batch
        self.attrs["in_features"] = in_features
        self.attrs["out_features"] = out_features
        

class ForwardLinearTensors(Tensors):
    """前传Linear算子 用到的张量
    """
    def __init__(self,input,
                        output,
                        weight,
                        ):
        super().__init__()
        self.tensors["input"] = input
        self.tensors["output"] = output
        self.tensors["weight"] = weight
        
        self.input = ["input",]
        self.output = ["output",]

class BackwardLinearTensors(Tensors):
    """反传Linear算子 用到的张量
    """
    def __init__(self,input_grad,
                        output_grad,
                        weight_grad,
                        input,
                        weight,
                        ):
        super().__init__()
        self.tensors["input_grad"] = input_grad
        self.tensors["output_grad"] = output_grad
        self.tensors["weight_grad"] = weight_grad
        self.tensors["input"] = input
        self.tensors["weight"] = weight
        
        self.input = ["output_grad",]
        self.output = ["input_grad",]

