# from executer.executer import bind
import torch
import torch.nn as nn
from compiler.utils.utils import padding_inside
import os
bind_table = dict()
def bind(operator):
    """ Bind execute function on operator

    Args:
        operator: The class name of operator to bind.

    Returns:
        decorate

    Examples:
        @bind(operator="ForwardRelu")
        def forward_relu(self):
            input = self.tensors.get_data("input")
            output = torch.relu(input)
            self.tensors.set_data("output",output)
            self.tensors.set_data("mask",input>0)
    """
    assert type(operator)==str
    def decorate(f):
        bind_table[operator] = f
        return f
    return decorate

@bind(operator="ForwardConv")
def forward_conv(self):

    #prepare attributes
    out_channels = self.attrs.get("out_channels")
    in_channels = self.attrs.get("in_channels")
    stride = self.attrs.get("stride")
    kernel_size = self.attrs.get("kernel_size")
    padding = self.attrs.get("padding")

    #prepare tensors
    input = self.tensors.get_data("input")
    weight = self.tensors.get_data("weight")

    #execute
    conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False)
    conv.weight = torch.nn.Parameter(weight)
    result = conv(input).detach()

    #write back
    self.tensors.set_data("output",result)

@bind(operator="BackwardConv")
def backward_conv(self):
    #prepare tensors
    input_grad = self.tensors.get("input_grad")
    input_width = input_grad.shape[3]
    weight = self.tensors.get_data("weight")
    output_grad = self.tensors.get_data("output_grad")
    output_grad = padding_inside(output_grad,self.attrs.get("stride")-1)

    weight = torch.flip(weight,[2,3])
    weight = torch.transpose(weight,0,1)
    out_channels,in_channels,kernel_size,_ = weight.shape
    padding = kernel_size - 1

    #execute
    conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False)#stride?padding?
    conv.weight = torch.nn.Parameter(weight)
    input_grad = conv(output_grad).detach()

    padding = self.attrs.get("padding")
    
    pad = input_width+2*padding-input_grad.shape[3]
    if pad>0:
        input_grad = torch.nn.functional.pad(input_grad,[0,pad,0,pad])

    if padding>0:
        input_grad = input_grad[:,:,padding:-padding,padding:-padding]

    #write back
    self.tensors.set_data("input_grad",input_grad)

@bind(operator="WGConv")
def weight_gradient_conv(self):
    #prepare tensors
    input = self.tensors.get_data("input")
    output_grad = self.tensors.get_data("output_grad")

    #prepare attributes
    out_channels, in_channels, kernel_size, _ = output_grad.shape
    padding = self.attrs.get("padding")
    stride = self.attrs.get("stride")

    #execute
    input = torch.transpose(input,0,1)
    output_grad = torch.transpose(output_grad,0,1)
    output_grad = padding_inside(output_grad,stride-1)
    _, _, kernel_size, _ = output_grad.shape
    conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False)#stride?padding?
    conv.weight = torch.nn.Parameter(output_grad)
    weight_grad = torch.transpose(conv(input).detach(),0,1)
    weight_grad = weight_grad[:,:,:self.attrs.get("kernel_size"),:self.attrs.get("kernel_size")]
    
    #write back
    self.tensors.set_data("weight_grad",weight_grad)


@bind(operator="WUConv")
def weight_update_conv(conv):
    pass

@bind(operator="ForwardSoftmax")
def forward_softmax(self):
    input = self.tensors.get_data("input")
    self.tensors.set_data("output",input)

@bind(operator="BackwardSoftmax")
def backward_softmax(self):
    output_grad = self.tensors.get_data("output_grad")
    self.tensors.set_data("input_grad",output_grad)

@bind(operator="ForwardEntropy")
def forward_entropy(self):
    input = self.tensors.get_data("input")
    result = torch.sum(input)
    self.tensors.set_data("loss",result)

@bind(operator="BackwardEntropy")
def backward_entropy(self):
    loss = self.tensors.get_data("loss")
    input_grad = self.tensors.get("input_grad")
    result = torch.ones(input_grad.shape)*1.00
    self.tensors.set_data("input_grad",result)

@bind(operator="ForwardFlatten")
def forward_flatten(self):
    input = self.tensors.get_data("input")
    result = torch.flatten(input,1)
    self.tensors.set_data("output",result)

@bind(operator="BackwardFlatten")
def backward_flatten(self):
    output_grad = self.tensors.get_data("output_grad")
    input_grad = self.tensors.get("input_grad")
    result = output_grad.reshape(input_grad.shape)
    self.tensors.set_data("input_grad",result)

@bind(operator="ForwardRelu")
def forward_relu(self):
    input = self.tensors.get_data("input")
    output = torch.relu(input)
    mask = input>0
    self.tensors.set_data("output",output)
    self.tensors.set_data("mask",mask)

@bind(operator="BackwardRelu")
def backward_relu(self):
    output_grad = self.tensors.get_data("output_grad")
    mask = self.tensors.get_data("mask")
    input_grad = torch.mul(output_grad,mask)
    self.tensors.set_data("input_grad",input_grad)

@bind(operator="ForwardAdd")
def forward_add(self):
    input1 = self.tensors.get_data("input1")
    input2 = self.tensors.get_data("input2")
    output = input1 + input2
    self.tensors.set_data("output",output)

@bind(operator="BackwardSplit")
def backward_split(self):
    output_grad1 = self.tensors.get_data("output_grad1")
    output_grad2 = self.tensors.get_data("output_grad2")
    input_grad = output_grad1 + output_grad2
    self.tensors.set_data("input_grad",input_grad)

@bind(operator="ForwardBatchnorm")
def forward_batchnorm(self):
    # prepare attributes
    num_features = self.attrs.get("num_features")
    affine = self.attrs.get("affine")

    # prepare tensors
    input = self.tensors.get_data("input")
    mean = self.tensors.get_data("mean")
    std = self.tensors.get_data("std")

    # compute
    output = torch.transpose(input,1,3)
    output = (output-mean)/std

    if affine:
        alpha = self.tensors.get_data("alpha")
        beta = self.tensors.get_data("beta")
        output = torch.mul(output,alpha) + beta
    output = torch.transpose(output,1,3)
    #     bn.weight = nn.Parameter(alpha)
    #     bn.bias = nn.Parameter(beta)

    # output = bn(input).detach()


    #save back
    self.tensors.set_data("output",output)

@bind(operator="BackwardBatchnorm")
def backward_batchnorm(self):
    # prepare attributes
    num_features = self.attrs.get("num_features")
    affine = self.attrs.get("affine")

    # prepare tensors
    output_grad = self.tensors.get_data("output_grad")
    std = self.tensors.get_data("std")

    # compute
    diff = 1 / std
    if affine:
        alpha = self.tensors.get_data("alpha")
        diff = torch.mul(alpha,diff)
    output_grad = torch.transpose(output_grad,1,3)
    input_grad = torch.mul(output_grad,diff)
    input_grad = torch.transpose(input_grad,1,3)

    #save back
    self.tensors.set_data("input_grad",input_grad)
    

@bind(operator="ForwardMaxpool")
def forward_maxpool(self):
    kernel_size = self.attrs.get("kernel_size")
    padding = self.attrs.get("padding")
    stride = self.attrs.get("stride")
    assert (kernel_size==2 and stride==2) or (kernel_size==4 and stride==4)
    assert padding==0
    input = self.tensors.get_data("input")
    output,indices = torch.nn.functional.max_pool2d(input, kernel_size,return_indices=True)
    
    batch,channel,height,width = input.shape
    mask = torch.zeros(batch,channel,height*width).to(input.device,input.dtype)
    indices = indices.reshape(batch,channel,-1)
    for b in range(batch):
        for c in range(channel):
            mask[b][c][indices[b][c]] = 1
    mask = mask.reshape(batch,channel,height,width)

    self.tensors.set_data("output",output)
    self.tensors.set_data("mask",mask)

@bind(operator="BackwardMaxpool")
def backward_maxpool(self):
    kernel_size = self.attrs.get("kernel_size")
    output_grad = self.tensors.get_data("output_grad")
    batch,channel,height,width = output_grad.shape

    mask = self.tensors.get_data("mask")

    input_grad = torch.cat([output_grad]*(kernel_size*kernel_size),dim=3)
    input_grad = input_grad.reshape(batch,channel,height*kernel_size,kernel_size,width)
    input_grad = torch.transpose(input_grad,3,4)#.contiguous()
    input_grad = input_grad.reshape(batch,channel,kernel_size*height,kernel_size*width)

    input_grad = torch.mul(input_grad,mask)
    self.tensors.set_data("input_grad",input_grad)

@bind(operator="ForwardLinear")
def forward_linear(self):
    out_features = self.attrs.get("out_features")
    in_features = self.attrs.get("in_features")
    input = self.tensors.get_data("input")
    weight = self.tensors.get_data("weight")

    linear = nn.Linear(in_features,out_features,bias=False)
    linear.weight = torch.nn.Parameter(weight)
    output = linear(input).detach()
    self.tensors.set_data("output",output)

@bind(operator="BackwardLinear")
def backward_linear(self):
    weight = self.tensors.get_data("weight")
    output_grad = self.tensors.get_data("output_grad")
    input_grad = torch.matmul(output_grad,weight)
    self.tensors.set_data("input_grad",input_grad)
