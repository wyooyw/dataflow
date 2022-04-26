import numpy as np
from test_op.communicate.S2TrainExample import S2TrainInterface
# from test_op.communicate.S2TrainInterface import S2TrainInterface
import test_op.op.utils as utils
import torch.nn as nn
from torch.autograd import Function


class S2TrainConv2dFunction(Function):
    """
    将卷积的前传、反传包装为torch.autograd.Function
    可以直接使用pytorch的自动求导来进行前传和反传
    这个函数会进一步封装到dataflow.op.S2TrainConv2d里
    """
    @staticmethod
    def forward(ctx, weight, input, **kwargs):
        ctx.save_for_backward(input, weight)
        return conv_forward(input, weight, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        weight = ctx.saved_tensors[1]
        return conv_backward(grad_output=grad_output,
                             input=input,
                             weight=weight)


def conv_forward(input: np.ndarray, weight: np.ndarray, **kwargs):
    """
    卷积前传
    完成分块，把分块后的算子发送到S2Train上面，并等待其完成计算
    """
    # TODO:把之前写的dataflow.split.split_conv拿到这里做分块
    output = S2TrainInterface.execute_conv(input, weight, **kwargs)
    return output


def conv_backward(output_grad: np.ndarray, input: np.ndarray,
                  weight: np.ndarray):
    """
    卷积反传
    分别计算输入梯度和权重梯度
    TODO:分块
    """
    input_grad = calc_input_grad(output_grad=output_grad, weight=weight)
    weight_grad = calc_weight_grad(output_grad=output_grad, input=input)
    return input_grad, weight_grad


def calc_input_grad(output_grad, weight):
    """
    计算输入梯度
    完成分块，把分块后的算子发送到S2Train上面，并等待其完成计算
    TODO:分块
    """
    weight = utils.reverse(weight)
    weight = utils.switch_batch_and_channel(weight)
    out_channels, in_channels, kernel_size, _ = weight.shape
    padding = kernel_size - 1

    input_grad = S2TrainInterface.execute_conv(input=output_grad,
                                               weight=weight,
                                               in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               padding=padding,
                                               bias=False)
    return input_grad


def calc_weight_grad(output_grad, input):
    """
    计算权重梯度
    完成分块，把分块后的算子发送到S2Train上面，并等待其完成计算
    TODO:分块
    """
    output_grad = utils.switch_batch_and_channel(output_grad)
    input = utils.switch_batch_and_channel(input)

    out_channels, in_channels, kernel_size, _ = output_grad.shape
    weight_grad = S2TrainInterface.execute_conv(input=input,
                                                weight=output_grad,
                                                in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                bias=False)

    weight_grad = utils.switch_batch_and_channel(weight_grad)
    return weight_grad


def HelloWorld():
    print("hwlloworld")
