from compiler.generate.operator import Operator,OperatorType
from compiler.utils.unique_class_name import unique_class_name
import compiler.utils.utils as utils
from compiler.generate.op.attrs.fuse_conv_relu_attrs import ForwardFuseConvReluAttrs
from compiler.generate.op.tensors.fuse_conv_relu_tensors import ForwardFuseConvReluTensors
from queue import Queue
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

    def refresh_tensor_shape_limit(self):
        """根据tensor之间的制约，更新对tensor的shape的限制
        """
        conv_input = self.conv.tensors.get("input")
        conv_output = self.conv.tensors.get("output")
        relu_input = self.relu.tensors.get("input")
        relu_output = self.relu.tensors.get("output")

        #原始shape
        conv_input_shape = conv_input.shape
        conv_output_shape = conv_output.shape
        relu_input_shape = relu_input.shape
        relu_output_shape = relu_output.shape

        #原始shape_limit
        conv_input_shape_lim = conv_input.shape_limit
        conv_output_shape_lim = conv_output.shape_limit
        relu_input_shape_lim = relu_input.shape_limit
        relu_output_shape_lim = relu_output.shape_limit

        #重新调整shape_limit
        relu_output_shape_lim = utils.min(relu_output_shape,relu_output_shape_lim)
        relu_input_shape_lim = utils.min(relu_input_shape,relu_input_shape_lim,relu_output_shape_lim)
        conv_output_shape_lim = utils.min(conv_output_shape,conv_output_shape_lim,relu_input_shape_lim)
        conv_input_shape_lim = utils.min(conv_input_shape,conv_input_shape_lim,
                                        ForwardConv2d.get_in_shape_by_out_shape(conv_output_shape_lim,self.conv.attrs))
        conv_output_shape_lim = ForwardConv2d.get_out_shape_by_in_shape(conv_input_shape_lim,self.conv.attrs)
        relu_input_shape_lim = conv_output_shape_lim
        relu_output_shape_lim = relu_input_shape_lim

        #用新的shape_limit替换旧的
        conv_input.shape_limit = conv_input_shape_lim
        conv_output.shape_limit = conv_output_shape_lim
        relu_input.shape_limit = relu_input_shape_lim
        relu_output.shape_limit = relu_output_shape_lim

    def split(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            split_instance = queue.get()

            #1.分割out_channel
            conv_weight = split_instance.conv.tensors.get("weight")
            out_channel = conv_weight.shape[0]
            out_channel_limit = conv_weight.shape_limit[0]
            if out_channel > out_channel_limit:
                for idx in range(0,out_channel+1,out_channel_limit):
                    op = copy.copy(split_instance)
                    op.conv.attrs.set("out_channels",out_channel)
                    #分块后下标的开始和结束
                    out_channel_start = idx
                    out_channel_stop = min(idx+out_channel_limit,out_channel)
                    #修改conv的kernel
                    conv_kernel_tensor = op.conv.tensors.get("kernel")
                    conv_kernel_tensor = conv_kernel_tensor[out_channel_start:out_channel_stop,:,:,:]
                    op.conv.tensors.set("kernel",kernel_tensor)
                    #修改conv的output
                    conv_output_tensor = op.conv.tensors.get("output")
                    conv_output_tensor = conv_output_tensor[:,out_channel_start:out_channel_stop,:,:]
                    op.conv.tensors.set("output",conv_output_tensor)
                    #修改relu的input
                    relu_input_tensor = op.relu.tensors.get("input")
                    relu_input_tensor = relu_input_tensor[:,out_channel_start:out_channel_stop,:,:]
                    op.relu.tensors.set("input",relu_input_tensor)
                    #修改relu的mask
                    relu_mask_tensor = op.relu.tensors.get("mask")
                    relu_mask_tensor = relu_mask_tensor[:,out_channel_start:out_channel_stop,:,:]
                    op.relu.tensors.set("mask",relu_mask_tensor)
                    #修改relu的output
                    relu_output_tensor = op.relu.tensors.get("output")
                    relu_output_tensor = relu_output_tensor[:,out_channel_start:out_channel_stop,:,:]
                    op.relu.tensors.set("output",relu_output_tensor)
                    #加到计算图上
                    op.connect_predecessor(split_instance.predecessor)
                    op.connect_successor(split_instance.successor)
                    #加到队列里，用于后续在其他维度做分割
                    queue.put(op)
                #从计算图里删掉自己
                split_instance.disconnect_predecessor(split_instance.predecessor)
                split_instance.disconnect_successor(split_instance.successor)
                continue
            
            #2.分割batch_size
            conv_input = split_instance.conv.tensors.get("input")
            batch_size = conv_input.shape[0]
            batch_size_limit = conv_input.shape_limit[0]
            if batch_size > batch_size_limit:
                for idx in range(0,batch_size+1,batch_size_limit):
                    op = copy.copy(split_instance)
                    #分块后下标的开始和结束
                    batch_size_start = idx
                    batch_size_stop = min(idx+batch_size_limit,batch_size)
                    #修改conv的kernel
                    conv_input_tensor = op.conv.tensors.get("input")
                    conv_input_tensor = conv_input_tensor[batch_size_start:batch_size_stop,:,:,:]
                    op.conv.tensors.set("input",conv_input_tensor)
                    #修改conv的output
                    conv_output_tensor = op.conv.tensors.get("output")
                    conv_output_tensor = conv_output_tensor[batch_size_start:batch_size_stop,:,:,:]
                    op.conv.tensors.set("output",conv_output_tensor)
                    #修改relu的input
                    relu_input_tensor = op.relu.tensors.get("input")
                    relu_input_tensor = relu_input_tensor[batch_size_start:batch_size_stop,:,:,:]
                    op.relu.tensors.set("input",relu_input_tensor)
                    #修改relu的mask
                    relu_mask_tensor = op.relu.tensors.get("mask")
                    relu_mask_tensor = relu_mask_tensor[batch_size_start:batch_size_stop,:,:,:]
                    op.relu.tensors.set("mask",relu_mask_tensor)
                    #修改relu的output
                    relu_output_tensor = op.relu.tensors.get("output")
                    relu_output_tensor = relu_output_tensor[batch_size_start:batch_size_stop,:,:,:]
                    op.relu.tensors.set("output",relu_output_tensor)
                    #加到计算图上
                    op.connect_predecessor(split_instance.predecessor)
                    op.connect_successor(split_instance.successor)
                    #加到队列里，用于后续在其他维度做分割
                    queue.put(op)
                #从计算图里删掉自己
                split_instance.disconnect_predecessor(split_instance.predecessor)
                split_instance.disconnect_successor(split_instance.successor)
                continue

            #3.分割width和height
            #weight和height分一次分割，还是分两次？
            #分到最后剩的不是正方形，需要补领吗？补领的话在哪里补？
            
            

        #最后，从计算图里删掉自己
        split_instance.disconnect_predecessor(split_instance.predecessor)
        split_instance.disconnect_successor(split_instance.successor)

    def 


    @classmethod
    def split_width(self):
        
        
# class BackwardFuseConvRelu(Operator):
#     def __init__(self,attrs:BackwardFuseConvReluAttrs,tensors:BackwardFuseConvReluTensors):
#         super().__init__(type=OperatorType.BACKWARD_FUSE_CONV_RELU,
#                         attrs=attrs,
#                         tensors=tensors,
#                         name=unique_class_name(self))