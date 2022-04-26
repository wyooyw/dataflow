from compiler.generate.operator import Operator,OperatorType
from compiler.utils.unique_class_name import unique_class_name
from compiler.generate.op.attrs.fuse_conv_relu_attrs import ForwardFuseConvReluAttrs
from compiler.generate.op.tensors.fuse_conv_relu_tensors import ForwardFuseConvReluTensors
class ForwardFuseConvRelu(Operator):
    def __init__(self,attrs:ForwardFuseConvReluAttrs,tensors:ForwardFuseConvReluTensors):
        super().__init__(type=OperatorType.FORWARD_FUSE_CONV_RELU,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self))

    @classmethod
    def merge(self,conv,relu):
        """将ForwardConv2d和ForwardRelu合并为ForwardFuseConvRelu
        """
        
        #合并attr
        conv_attrs = conv.attrs
        relu_attrs = relu.attrs
        attrs = ForwardFuseConvReluAttrs(conv_in_channels=conv_attrs.get("in_channels"),
                                        conv_out_channels=conv_attrs.get("out_channels"),
                                        conv_kernel_size=conv_attrs.get("kernel_size"),
                                        conv_padding=conv_attrs.get("padding"))
        
        #合并tensor
        conv_tensors = conv.tensors
        relu_tensors = relu.tensors
        tensors = ForwardFuseConvReluTensors(conv_weight=conv_tensors.get("weight"),
                                            conv_input=conv_tensors.get("input"),
                                            conv_output=conv_tensors.get("output"),
                                            relu_mask=relu_tensors.get("mask"),
                                            relu_input=relu_tensors.get("input"),
                                            relu_output=relu_tensors.get("output"))
        
        #创建新算子
        forward_fuse_conv_relu = ForwardFuseConvRelu(attrs=attrs,
                                                    tensors=tensors)

        #合并后继节点
        origin = set([conv,relu])
        successors = (conv.successor | relu.successor) - origin
        for successor in successors:
            successor.remove_predecessor(*origin)
            successor.connect_predecessor(forward_fuse_conv_relu)

        #合并前驱节点
        predecessors = (conv.predecessor | relu.predecessor) - origin
        for predecessor in predecessors:
            predecessor.remove_successor(*origin)
            predecessor.connect_successor(forward_fuse_conv_relu)

        #替换网络中的算子
        net = conv.net
        net.remove_operator(*origin)
        net.add_operator(forward_fuse_conv_relu)

        return forward_fuse_conv_relu

# class BackwardFuseConvRelu(Operator):
#     def __init__(self,attrs:BackwardFuseConvReluAttrs,tensors:BackwardFuseConvReluTensors):
#         super().__init__(type=OperatorType.BACKWARD_FUSE_CONV_RELU,
#                         attrs=attrs,
#                         tensors=tensors,
#                         name=unique_class_name(self))