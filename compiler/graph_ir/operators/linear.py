from compiler.graph_ir import Dual,Operator,OperatorType,Attrs
from compiler.graph_ir.attrs.linear_attrs import *
from compiler.graph_ir.tensors.linear_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn
import torch
from simulator.memory import Memory
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits

class DualLinear(Dual):
    def __init__(self,in_shape,
                        in_features,
                        out_features):
        super().__init__()
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
                                                    output_grad=output_grad,
                                                    input_grad=input_grad)
        weight_gradient_tensors = WGLinearTensors(input=input,
                                                output_grad=output_grad,
                                                weight_grad=weight_grad)
        weight_update_tensors = WULinearTensors(weight=weight,
                                                weight_grad=weight_grad)
        
        #定义op
        self.forward = ForwardLinear(attrs=forward_linear_attrs,
                                        tensors=forward_linear_tensors,
                                        in_shape=in_shape,
                                        out_shape=out_shape)
        self.backward = BackwardLinear(attrs=backward_linear_attrs,
                                        tensors=backward_linear_tensors,
                                        in_shape=out_shape,
                                        out_shape=in_shape)
        self.weight_gradient = WGLinear(attrs=Attrs(),
                                        tensors=weight_gradient_tensors)
        self.weight_update = WULinear(attrs=Attrs(),
                                        tensors=weight_update_tensors)

    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualLinear(in_shape=in_shape,
                        in_features=module.in_features,
                        out_features=module.out_features)
        dual.forward.get_tensors().tensors["weight"].storage.data = module.weight.detach().numpy()
        return dual

class ForwardLinear(Operator):
    """前传算子
    """
    def __init__(self,attrs:ForwardLinearAttrs,tensors:ForwardLinearTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
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
    
    def sim_run(self):
        out_features = self.attrs.get("out_features")
        in_features = self.attrs.get("in_features")
        input = self.tensors.get("input")
        weight = self.tensors.get("weight")
        output = self.tensors.get("output")

        input = Memory().get(input.addr)
        weight = Memory().get(weight.addr)
        linear = nn.Linear(in_features,out_features,bias=False)
        linear.weight = torch.nn.Parameter(weight)
        Memory().set(output.addr,linear(input).detach())

    def to_instr(self):
        instruction = Instruction(name=self.name, init_data={
            "net_type": "resnet",
            "stage": "forward",
            "op_type": "linear",
            "stride": 1,
            "padding": False,
            "relu": False,
            "maxpool": False,
            "kernel_size": 1,
            "add": False,
            "bn": False,
            "part_sum": False,
            "softmax": False
        },pad_to=128)
        instruction.set("in_feature", int_to_bits(self.tensors.get("input").index,9).to01(),use_bits=True)
        instruction.set("weight", int_to_bits(self.tensors.get("weight").index,9).to01(),use_bits=True)
        instruction.set("output", int_to_bits(self.tensors.get("output").index,9).to01(),use_bits=True)
        return instruction

class BackwardLinear(Operator):
    """反传算子（特征图梯度算子）
    """
    def __init__(self,attrs:BackwardLinearAttrs,tensors:BackwardLinearTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

    def sim_run(self):
        weight = self.tensors.get("weight")
        output_grad = self.tensors.get("output_grad")
        input_grad = self.tensors.get("input_grad")

        weight = Memory().get(weight.addr)
        output_grad = Memory().get(output_grad.addr)
        # output_grad = torch.transpose(output_grad,0,1)

        Memory().set(input_grad.addr,torch.matmul(output_grad,weight))
    def to_instr(self):
        instruction = Instruction(name=self.name,init_data={
            "net_type":"resnet",
            "stage":"backward",
            "op_type":"linear",
            "stride":1,
            "padding":False,
            "relu": False,
            "maxpool": False,
            "kernel_size":1,
            "add":False,
            "bn":False,
            "part_sum":False,
            "softmax":False
        },pad_to=128)
        instruction.set("output_grad",int_to_bits(self.tensors.get("output_grad").index,9).to01(),use_bits=True)
        instruction.set("weight",int_to_bits(self.tensors.get("weight").index,9).to01(),use_bits=True)
        instruction.set("input_grad",int_to_bits(self.tensors.get("input_grad").index,9).to01(),use_bits=True)
        return instruction
class WGLinear(Operator):
    """权重梯度算子
    """
    def __init__(self,attrs:Attrs,tensors:WGLinearTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.WEIGHT_GRADIENT,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    def sim_run(self):
        input = self.tensors.get("input")
        output_grad = self.tensors.get("output_grad")
        weight_grad = self.tensors.get("weight_grad")

        input = Memory().get(input.addr)
        output_grad = Memory().get(output_grad.addr)
        output_grad = torch.transpose(output_grad,0,1)

        Memory().set(weight_grad.addr,torch.matmul(output_grad,input))
    
    def to_instr(self):
        instruction = Instruction(name=self.name, init_data={
            "net_type": "resnet",
            "stage": "WG",
            "op_type": "linear",
            "stride": 1,
            "padding": False,
            "relu": False,
            "maxpool": False,
            "kernel_size": 1,
            "add": False,
            "bn": False,
            "part_sum": False,
            "softmax": False
        },pad_to=128)
        instruction.set("output_grad", int_to_bits(self.tensors.get("output_grad").index,9).to01(),use_bits=True)
        instruction.set("weight_grad", int_to_bits(self.tensors.get("weight_grad").index,9).to01(),use_bits=True)
        instruction.set("input", int_to_bits(self.tensors.get("input").index,9).to01(),use_bits=True)
        return instruction

class WULinear(Operator):
    """权重更新算子
    """
    def __init__(self,attrs:Attrs,tensors:WULinearTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.WEIGHT_UPDATE,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

    def to_instr(self):
        instruction = Instruction(name=self.name, init_data={
            "net_type": "resnet",
            "stage": "WU",
            "op_type": "linear",
            "stride": 1,
            "padding": False,
            "relu": False,
            "maxpool": False,
            "kernel_size": 1,
            "add": False,
            "bn": False,
            "part_sum": False,
            "softmax": False
        },pad_to=128)
        instruction.set("weight", int_to_bits(self.tensors.get("weight").index,9).to01(),use_bits=True)
        instruction.set("weight_grad", int_to_bits(self.tensors.get("weight_grad").index,9).to01(),use_bits=True)
        return instruction