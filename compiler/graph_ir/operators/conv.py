from compiler.graph_ir import Dual,Operator,OperatorType,Attrs
from compiler.graph_ir.attrs.conv_attrs import *
from compiler.graph_ir.tensors.conv_tensors import *
from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.utils.pointer import Pointer
from compiler.utils.utils import padding_inside
from compiler.utils.unique_class_name import unique_class_name
from backends.sparse_train.target_code.instruction import Instruction
from compiler.utils.utils import int_to_bits
import torch.nn as nn
import torch
from simulator.memory import Memory

class DualConv(Dual):
    def __init__(self,in_shape,
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=0,
                        stride=1):
        # in_batch,
        # in_width,
        # in_height,
        super().__init__()
        in_batch,in_channels,in_width,in_height = in_shape

        forward_attrs = ForwardConvAttrs(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride)
        backward_attrs = BackwardConvAttrs(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                stride=stride)
        #定义张量
        weight = MemoryManager().allocWeight(shape=(out_channels,in_channels,kernel_size,kernel_size))
        weight_grad = MemoryManager().allocWeightGrad(shape=(out_channels,in_channels,kernel_size,kernel_size))
        input = MemoryManager().allocActivation(shape=(in_batch,in_channels,in_height,in_width))
        input_grad = MemoryManager().allocFeatureGrad(shape=(in_batch,in_channels,in_height,in_width))
        #TODO output_size把stride考虑进来，调用ForwardConv的get_out_shape_by_in_shape
        out_shape = ForwardConv.get_out_shape_by_in_shape(in_shape,forward_attrs)
        output_grad = MemoryManager().allocFeatureGrad(shape=out_shape)
        output = MemoryManager().allocActivation(shape=out_shape)

        forward_tensors = ForwardConvTensors(weight=weight,
                                                    input=input,
                                                    output=output)
        backward_tensors = BackwardConvTensors(weight=weight,
                                                    output_grad=output_grad,
                                                    input_grad=input_grad)
        weight_gradient_tensors = WGConvTensors(input=input,
                                                output_grad=output_grad,
                                                weight_grad=weight_grad)
        weight_update_tensors = WUConvTensors(weight=weight,
                                                weight_grad=weight_grad)

        #定义op
        self.forward = ForwardConv(attrs=forward_attrs,
                                        tensors=forward_tensors,
                                        in_shape=in_shape,
                                        out_shape=out_shape)
        self.backward = BackwardConv(attrs=backward_attrs,
                                        tensors=backward_tensors,
                                        in_shape=out_shape,
                                        out_shape=in_shape)
        self.weight_gradient = WGConv(attrs=forward_attrs,
                                        tensors=weight_gradient_tensors)
        self.weight_update = WUConv(attrs=Attrs(),
                                        tensors=weight_update_tensors)

    @classmethod
    def from_torch_module(cls,in_shape,module):
        dual = DualConv(in_shape=in_shape,
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size[0],
                        stride=module.stride[0],
                        padding=module.padding[0])
        dual.forward.get_tensors().tensors["weight"].storage.data = module.weight.detach()
        return dual

class ForwardConv(Operator):
    """前传算子
    """
    def __init__(self,attrs:ForwardConvAttrs,tensors:ForwardConvTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.FORWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)

    @classmethod
    def get_in_shape_by_out_shape(cls,out_shape,attr:ForwardConvAttrs):
        """根据输出shape和算子attr,计算输入shape
        """
        assert type(out_shape)==tuple or type(out_shape)==list
        assert len(out_shape)==4
        out_batch,out_channel,out_height,out_width = out_shape
        in_channels = attr.get("in_channels")
        stride = attr.get("stride")
        kernel_size = attr.get("kernel_size")
        padding = attr.get("padding")
        """
        实际上,in_width的范围是[out_width*stride+kernel_size-2*padding-stride,out_width*stride+kernel_size-2*padding-1],
        这里取了下确界
        推导见https://www.wolai.com/2xMkPwJEPuiYAzuQN9fDpx
        """
        in_width = out_width * stride + kernel_size - 2 * padding - stride

        """
        宽高相等
        """
        in_height = in_width

        in_channel = in_channels

        in_batch = out_batch

        return (in_batch,in_channel,in_height,in_width)

    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr:ForwardConvAttrs):
        """根据输入shape和算子attr,计算输出shape
        """
        assert type(in_shape)==tuple or type(in_shape)==list,f"Type if in_shape should be tuple or list, but got {in_shape}"
        assert len(in_shape)==4
        in_batch,in_channel,in_height,in_width = in_shape
        out_channels = attr.get("out_channels")
        stride = attr.get("stride")
        kernel_size = attr.get("kernel_size")
        padding = attr.get("padding")
        
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1

        """
        宽高相等
        """
        out_height = out_width

        out_channel = out_channels

        out_batch = in_batch

        return (out_batch,out_channel,out_height,out_width)
    
    # def sim_run(self):
    #     out_channels = self.attrs.get("out_channels")
    #     in_channels = self.attrs.get("in_channels")
    #     stride = self.attrs.get("stride")
    #     kernel_size = self.attrs.get("kernel_size")
    #     padding = self.attrs.get("padding")
    #     input = self.tensors.get("input")
    #     weight = self.tensors.get("weight")
    #     output = self.tensors.get("output")

    #     input = Memory().get(input.addr)
    #     weight = Memory().get(weight.addr)
    #     conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False)
    #     conv.weight = torch.nn.Parameter(weight)
    #     Memory().set(output.addr,conv(input).detach())
        

class BackwardConv(Operator):
    """反传算子（特征图梯度算子）
    """
    def __init__(self,attrs:BackwardConvAttrs,tensors:BackwardConvTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.BACKWARD,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    # def sim_run(self):
    #     input_grad = self.tensors.get("input_grad")
    #     input_width = input_grad.shape[3]
    #     weight = self.tensors.get("weight")
    #     output_grad = self.tensors.get("output_grad")

    #     weight = Memory().get(weight.addr)
    #     output_grad = Memory().get(output_grad.addr)
    #     output_grad = padding_inside(output_grad,self.attrs.get("stride")-1)

    #     weight = torch.flip(weight,[2,3])
    #     weight = torch.transpose(weight,0,1)
    #     out_channels,in_channels,kernel_size,_ = weight.shape
    #     padding = kernel_size - 1
    #     input_grad_addr = input_grad.addr

    #     conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False)#stride?padding?
    #     conv.weight = torch.nn.Parameter(weight)
    #     input_grad = conv(output_grad).detach()
    #     # print("input_grad:",input_grad)
    #     padding = self.attrs.get("padding")
    #     # import ipdb
    #     # ipdb.set_trace()
        
    #     pad = input_width+2*padding-input_grad.shape[3]
    #     if pad>0:
    #         input_grad = torch.nn.functional.pad(input_grad,[0,pad,0,pad])
    #         # print("input_grad_pad:",input_grad)

    #     if padding>0:
    #         input_grad = input_grad[:,:,padding:-padding,padding:-padding]

    #     Memory().set(input_grad_addr,input_grad)
    
    def to_instr(self):
        instruction = Instruction(name=self.name,init_data={
            "net_type":"resnet",
            "stage":"backward",
            "op_type":"conv",
            "stride":self.attrs.get("stride"),
            "padding":False,
            "relu": False,
            "maxpool": False,
            "kernel_size":self.attrs.get("kernel_size"),
            "add":False,
            "bn":False,
            "part_sum":False,
            "softmax":False
        },pad_to=128)
        instruction.set("output_grad",int_to_bits(self.tensors.get("output_grad").index,9).to01(),use_bits=True)
        instruction.set("weight",int_to_bits(self.tensors.get("weight").index,9).to01(),use_bits=True)
        instruction.set("input_grad",int_to_bits(self.tensors.get("input_grad").index,9).to01(),use_bits=True)
        return instruction

class WGConv(Operator):
    """权重梯度算子
    """
    def __init__(self,attrs:Attrs,tensors:WGConvTensors,in_shape=[],out_shape=[]):
        super().__init__(type=OperatorType.WEIGHT_GRADIENT,
                        attrs=attrs,
                        tensors=tensors,
                        name=unique_class_name(self),
                        in_shape=in_shape,
                        out_shape=out_shape)
    
    # def sim_run(self):

    #     input = self.tensors.get("input")
    #     weight_grad = self.tensors.get("weight_grad")
    #     output_grad = self.tensors.get("output_grad")
    #     out_channels, in_channels, kernel_size, _ = output_grad.shape
    #     padding = self.attrs.get("padding")
    #     stride = self.attrs.get("stride")

    #     input = Memory().get(input.addr)
    #     output_grad = Memory().get(output_grad.addr)
    #     input = torch.transpose(input,0,1)
    #     output_grad = torch.transpose(output_grad,0,1)

    #     output_grad = padding_inside(output_grad,stride-1)
    #     _, _, kernel_size, _ = output_grad.shape

    #     conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False)#stride?padding?
    #     conv.weight = torch.nn.Parameter(output_grad)
    #     weight_grad_addr = weight_grad.addr
    #     weight_grad = torch.transpose(conv(input).detach(),0,1)
    #     weight_grad = weight_grad[:,:,:self.attrs.get("kernel_size"),:self.attrs.get("kernel_size")]
    #     Memory().set(weight_grad_addr,weight_grad)

    def to_instr(self):
        instruction = Instruction(name=self.name, init_data={
            "net_type": "resnet",
            "stage": "WG",
            "op_type": "conv",
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

class WUConv(Operator):
    """权重更新算子
    """
    def __init__(self,attrs:Attrs,tensors:WUConvTensors,in_shape=[],out_shape=[]):
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
            "op_type": "conv",
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