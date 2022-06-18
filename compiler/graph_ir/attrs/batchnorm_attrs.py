from compiler.graph_ir import Attrs
class ForwardBatchnormAttrs(Attrs):
    def __init__(self,num_features,affine):
        super().__init__()
        self.attrs["num_features"] = num_features
        self.attrs["affine"] = affine

class BackwardBatchnormAttrs(Attrs):
    def __init__(self,num_features,affine):
        super().__init__()
        self.attrs["num_features"] = num_features
        self.attrs["affine"] = affine