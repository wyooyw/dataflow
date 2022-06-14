from compiler.graph_ir import Attrs
class ForwardLinearAttrs(Attrs):
    def __init__(self,in_features,
                        out_features):
        super().__init__()
        self.attrs["in_features"] = in_features
        self.attrs["out_features"] = out_features

class BackwardLinearAttrs(Attrs):
    def __init__(self,in_features,
                        out_features):
        super().__init__()
        self.attrs["in_features"] = in_features
        self.attrs["out_features"] = out_features