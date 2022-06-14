from compiler.graph_ir import Attrs
class ForwardMaxpoolAttrs(Attrs):
    def __init__(self,kernel_size,
                    padding,
                    stride):
        super().__init__()
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding
        self.attrs["stride"] = stride

class BackwardMaxpoolAttrs(Attrs):
    def __init__(self,kernel_size,
                    padding,
                    stride):
        super().__init__()
        self.attrs["kernel_size"] = kernel_size
        self.attrs["padding"] = padding
        self.attrs["stride"] = stride