from compiler.graph_ir import Attrs
class ForwardFuseConvReluAttrs(Attrs):
    def __init__(self,conv_in_channels,
                        conv_out_channels,
                        conv_kernel_size,
                        conv_padding=0):
        super().__init__()
        self.attrs["conv_in_channels"] = conv_in_channels
        self.attrs["conv_out_channels"] = conv_out_channels
        self.attrs["conv_kernel_size"] = conv_kernel_size
        self.attrs["conv_padding"] = conv_padding

class BackwardFuseConvReluAttrs(Attrs):
    def __init__(self,conv_in_channels,
                        conv_out_channels,
                        conv_kernel_size,
                        conv_padding=0):
        super().__init__()
        self.attrs["conv_in_channels"] = conv_in_channels
        self.attrs["conv_out_channels"] = conv_out_channels
        self.attrs["conv_kernel_size"] = conv_kernel_size
        self.attrs["conv_padding"] = conv_padding