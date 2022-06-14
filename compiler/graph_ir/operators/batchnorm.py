from compiler.graph_ir import Dual,Operator,OperatorType,Attrs
from compiler.graph_ir.attrs.batchnorm_attrs import *
from compiler.graph_ir.tensors.batchnorm_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.unique_class_name import unique_class_name
import torch.nn as nn
import torch
from simulator.memory import Memory

class DualBatchnorm(Dual):
    def __init__(self,in_shape):
        super().__init__()
        in_batch,in_channels,in_height,in_width = in_shape
        #定义张量
        input = MemoryManager().allocActivation(shape=in_shape)
        input_grad = MemoryManager().allocFeatureGrad(shape=in_shape)
        output = MemoryManager().allocActivation(shape=in_shape)
        output_grad = MemoryManager().allocFeatureGrad(shape=in_shape)

        avg = MemoryManager().allocActivation(shape=(in_channels,))
        std = MemoryManager().allocActivation(shape=(in_channels,))
        alpha = MemoryManager().allocWeight(shape=(in_channels,))
        beta = MemoryManager().allocWeight(shape=(in_channels,))
        
        forward_tensors = ForwardBatchnormTensors(avg=avg,
                                                std=std,
                                                alpha=alpha,
                                                beta=beta,
                                                input=input,
                                                output=output)
        backward_tensors = BackwardBatchnormTensors(avg=avg,
                                                std=std,
                                                alpha=alpha,
                                                beta=beta,
                                                input_grad=input_grad,
                                                output_grad=output_grad)

        forward_attrs = ForwardBatchnormAttrs(num_features=in_channels)
        backward_attrs = BackwardBatchnormAttrs(num_features=in_channels)
        #定义op
        self.forward = ForwardBatchnorm(attrs=forward_attrs,
                                        tensors=forward_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
        self.backward = BackwardBatchnorm(attrs=backward_attrs,
                                        tensors=backward_tensors,
                                        in_shape=in_shape,
                                        out_shape=in_shape)
    
    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualBatchnorm(in_shape=in_shape)
        assert not (module.running_mean==None or module.running_var==None),"PyTorch BatchNorm2d doesn't have running_mean or running_var."
        assert not (module.weight==None or module.bias==None),"PyTorch BatchNorm2d doesn't have weight or bias."
        
        dual.forward.get_tensors().tensors["avg"].storage.data = module.running_mean.detach().numpy()
        #TODO 改一下std的名字，改成var
        dual.forward.get_tensors().tensors["std"].storage.data = module.running_var.detach().numpy()
        dual.forward.get_tensors().tensors["alpha"].storage.data = module.weight.detach().numpy()
        dual.forward.get_tensors().tensors["beta"].storage.data = module.bias.detach().numpy()
        return dual


class ForwardBatchnorm(Operator):
    def __init__(self,attrs:ForwardBatchnormAttrs,tensors:ForwardBatchnormTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    @classmethod
    def get_in_shape_by_out_shape(cls,out_shape,attr):
        return out_shape
    
    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr):
        return in_shape

    def sim_run(self):
        print(self.tensors.tensors.keys())
        num_features = self.attrs.get("num_features")
        input = self.tensors.get("input")
        output = self.tensors.get("output")
        avg = self.tensors.get("avg")
        var = self.tensors.get("std")
        alpha = self.tensors.get("alpha")
        beta = self.tensors.get("beta")

        input = Memory().get(input.addr)
        avg = Memory().get(avg.addr)
        var = Memory().get(var.addr)
        alpha = Memory().get(alpha.addr)
        beta = Memory().get(beta.addr)

        bn = nn.BatchNorm2d(num_features)
        bn.eval()
        bn.running_mean = avg
        bn.running_var = var
        bn.weight = nn.Parameter(alpha)
        bn.bias = nn.Parameter(beta)
        Memory().set(output.addr,bn(input).detach())
        

class BackwardBatchnorm(Operator):
    def __init__(self,attrs:BackwardBatchnormAttrs,tensors:BackwardBatchnormTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

    def sim_run(self):
        print(self.tensors.tensors.keys())
        num_features = self.attrs.get("num_features")
        # eps = self.attrs.get("eps")
        input_grad = self.tensors.get("input_grad")
        output_grad = self.tensors.get("output_grad")
        var = self.tensors.get("std")
        alpha = self.tensors.get("alpha")

        output_grad = Memory().get(output_grad.addr)
        var = Memory().get(var.addr)
        alpha = Memory().get(alpha.addr)
        # import ipdb
        # ipdb.set_trace()
        diff = torch.mul(alpha,1/torch.sqrt(var+1e-5))
        output_grad = torch.transpose(output_grad,1,3)
        result = torch.mul(output_grad,diff)
        result = torch.transpose(result,1,3)

        Memory().set(input_grad.addr,result.detach())