from compiler.generate.op.attrs.attrs import Attrs
class ForwardBatchnormAttrs(Attrs):
    def __init__(self,num_features):
        super().__init__()
        self.attrs["num_features"] = num_features

class BackwardBatchnormAttrs(Attrs):
    def __init__(self,num_features):
        super().__init__()
        self.attrs["num_features"] = num_features
