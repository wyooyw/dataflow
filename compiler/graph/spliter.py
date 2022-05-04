from compile.generate.op.fuse_conv_relu import ForwardFuseConvRelu
import math
CONV_MAX_IN_CHANNELS = 4
class Spliter(object):
    def __init__(self):
        pass
    def split(self,op):
        assert type(op)==ForwardFuseConvRelu
        in_channels = op.attrs.get("in_channels")
        if in_channels <= CONV_MAX_IN_CHANNELS:
            return
        split_in_channels = [CONV_MAX_IN_CHANNELS]*(math.floor(in_channels/CONV_MAX_IN_CHANNELS))
        conv_input = split_op.tensors.get("conv_input")
        conv_weight = split_op.tensors.get("conv_weight")
        conv_output = split_op.tensors.get("conv_output")
        split_ops = []
        add_op = ForwardInplaceAdd()
        for idx,item in enumerate(split_in_channels):
            split_op = copy.copy(op)
            split_conv_input = conv_input[:,idx*CONV_MAX_IN_CHANNELS:(idx+1)*CONV_MAX_IN_CHANNELS,:,:]
            split_conv_weight = conv_weight[:,idx*CONV_MAX_IN_CHANNELS:(idx+1)*CONV_MAX_IN_CHANNELS,:,:]
            split_conv_output = allocActivation(shape=conv_output.shape)
            split_relu_input = split_conv_output
            split_op.connect_successor(add_op) #计算完每个split_fuse_conv_relu后计算add
            add_op.tensors.set_("input")
            split_ops.append(split_op)
        add_op.tensors.set("output",)
        #设置add_op的后继节点为原来fuse_conv_relu的后继节点
        
        # self.tensors["conv_weight"] = conv_weight
        # self.tensors["conv_input"] = conv_input
        # self.tensors["conv_output"] = conv_output
        # self.tensors["relu_mask"] = relu_mask
        # self.tensors["relu_input"] = relu_input
        # self.tensors["relu_output"] = relu_output
