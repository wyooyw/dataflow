from compiler.generate.dual import Dual
from compiler.generate.operator import Operator,OperatorType
from compiler.generate.op.attrs.linear_attrs import *
from compiler.generate.op.tensors.linear_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn

class DualLinear(Dual):
    def __init__(self,in_shape,
                        in_features,
                        out_features):

        in_batch,in_features = in_shape

        forward_linear_attrs = ForwardLinearAttrs(in_features=in_features,
                                                out_features=out_features)
        backward_linear_attrs = BackwardLinearAttrs(in_features=in_features,
                                                out_features=out_features)

        out_shape = ForwardLinear.get_out_shape_by_in_shape(in_shape,forward_linear_attrs)
        #定义张量
        weight =MemoryManager().allocWeight(shape=(in_features,out_features))
        weight_grad =MemoryManager().allocWeightGrad(shape=(in_features,out_features))
        input = MemoryManager().allocActivation(shape=(in_batch,in_features))
        input_grad =MemoryManager().allocFeatureGrad(shape=(in_batch,in_features))
        output =MemoryManager().allocActivation(shape=(in_batch,out_features))
        output_grad =MemoryManager().allocFeatureGrad(shape=(in_batch,out_features))

        forward_linear_tensors = ForwardLinearTensors(weight=weight,
                                                    input=input,
                                                    output=output)
        backward_linear_tensors = BackwardLinearTensors(weight=weight,
                                                    input=input,
                                                    output_grad=output_grad,
                                                    weight_grad=weight_grad,
                                                    input_grad=input_grad)

        #定义op
        self.forward = ForwardLinear(attrs=forward_linear_attrs,
                                        tensors=forward_linear_tensors,
                                        in_shape=in_shape,
                                        out_shape=out_shape)
        self.backward = BackwardLinear(attrs=backward_linear_attrs,
                                        tensors=backward_linear_tensors,
                                        in_shape=out_shape,
                                        out_shape=in_shape)

    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualLinear(in_shape=in_shape,
                        in_features=module.in_features,
                        out_features=module.out_features)
        return dual

class ForwardLinear(Operator):
    def __init__(self,attrs:ForwardLinearAttrs,tensors:ForwardLinearTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD_LINEAR,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr:ForwardLinearAttrs):
        """根据输入shape和算子attr,计算输出shape
        """
        batch,in_features = in_shape
        out_shape = (batch,attr.get("out_features"))
        return out_shape

class BackwardLinear(Operator):
    def __init__(self,attrs:BackwardLinearAttrs,tensors:BackwardLinearTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD_LINEAR,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

