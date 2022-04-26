from compiler.generate.op.attrs.attrs import Attrs
class ForwardConvAttrs(Attrs):
    def __init__(self,in_channels,
                        out_channels,
                        kernel_size,
                        padding=0):
        super().__init__()
        self.attrs["in_channels"] = in_channels
        self.attrs["out_channels"] = out_channels
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding

class BackwardConvAttrs(Attrs):
    def __init__(self,in_channels,
                        out_channels,
                        kernel_size,
                        padding=0):
        super().__init__()
        self.attrs["in_channels"] = in_channels
        self.attrs["out_channels"] = out_channels
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding