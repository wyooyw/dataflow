from compiler.graph_ir import Attrs
class ForwardConvAttrs(Attrs):
    def __init__(self,in_channels,
                        out_channels,
                        kernel_size,
                        padding,
                        stride):
        super().__init__()
        self.attrs["in_channels"] = in_channels
        self.attrs["out_channels"] = out_channels
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding
        self.attrs["stride"] = stride

class BackwardConvAttrs(Attrs):
    def __init__(self,in_channels,
                        out_channels,
                        kernel_size,
                        padding,
                        stride):
        super().__init__()
        self.attrs["in_channels"] = in_channels
        self.attrs["out_channels"] = out_channels
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding
        self.attrs["stride"] = stride