
class ForwardFuseConvRelu(Operator):
    def __init__(self,attrs:ForwardFuseConvReluAttrs,tensors:ForwardFuseConvReluTensors):
        super().__init__(type=OperatorType.FORWARD_FUSE_CONV_RELU,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

    @classmethod
    def merge_from(self,conv:ForwardConv2d,relu:ForwardRelu):
        """将ForwardConv2d和ForwardRelu合并为ForwardFuseConvRelu
        """
        #合并attr和tensor
        conv_attrs = conv.attrs
        relu_attrs = relu.attrs
        attrs = ForwardFuseConvReluAttrs(conv_in_channels=conv_attrs.get("in_channels"),
                                        conv_out_channels=conv_attrs.get("out_channels"),
                                        conv_kernel_size=conv_attrs.get("kernel_size"),
                                        conv_padding=conv_attrs.get("padding"))

        conv_tensors = conv.tensors
        relu_tensors = relu.tensors
        tensors = ForwardFuseConvReluTensors(conv_weight=conv_tensors.get("weight"),
                                            conv_input=conv_tensors.get("input"),
                                            conv_output=conv_tensors.get("output"),
                                            relu_mask=conv_tensors.get("mask"),
                                            relu_input=conv_tensors.get("input"),
                                            relu_output=conv_tensors.get("output"))
        
        forward_fuse_conv_relu = ForwardFuseConvRelu(attrs=attrs,
                                                    tensors=tensors)

        #合并后继节点
        conv_successor = set(conv.successor) - set([conv,relu])
        relu_successor = set(relu.successor) - set([conv,relu])
        # successors = set([*conv_successor,*relu_successor])
        for successor in conv_successor:
            successor.remove_predecessor(conv)
            forward_fuse_conv_relu.add_successor(successor)

        #TODO 是否需要合并前驱节点？
        conv_predecessor = conv.predecessor
        relu_predecessor = relu.predecessor
        predecessors = set([*conv_predecessor,*relu_predecessor]) - set([conv,relu])
        for predecessor in predecessors:
            forward_fuse_conv_relu.add_predecessor(predecessor)

        return forward_fuse_conv_relu

class BackwardFuseConvRelu(Operator):
    def __init__(self,attrs:BackwardFuseConvReluAttrs,tensors:BackwardFuseConvReluTensors):
        super().__init__(type=OperatorType.BACKWARD_FUSE_CONV_RELU,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))