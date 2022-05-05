# S2trainDataflow

### 介绍

毕设，针对S2Train芯片的数据流映射。主要包含两个功能：

1.在S2Train芯片设计阶段，为芯片提供单个算子和整个网络的测试数据，包括测试数据生成、数据格式转换、与仿真进程通信、计算结果检查等工作。

2.在S2Train芯片完成之后，为控制芯片的FPGA提供输入数据和网络结构信息，主要工作是对输入数据做格式转换。

#### 格式转换:

dataflow/format下的代码用于进行权重、输入的格式转换

TODO:输出的格式转换

#### 与仿真进程通信

dataflow/communicate下的代码用于与仿真进程进行通信

外部通过调用dataflow/communicate/S2TrainInterface.py来发送任务给S2Train仿真进程

目前还没确定如何通信，确定后具体实现里面的代码



#### S2Train算子

一个S2Train算子在前传、反传的过程中，完成如下事情：

1.对输入数据及权重的分块、格式转换

2.把数据发送到仿真进程上，等待仿真进程计算得到结果

3.将计算结果转换回普通张量格式，并将分块后的多个计算结果进行组合

例如，现在代码里的dataflow.op.S2TrainConv2d（大致结构有了，还未完善）

#### 单个算子的测试

以卷积算子为例，准备好torch.nn.Conv2d和dataflow.op.S2TrainConv2d，为它们俩赋予相同的输入、权重和属性，分别执行。

torch.nn.Conv2d会直接计算出正确结果

dataflow.op.S2TrainConv2d会将输入和权重进行格式转换，发送到S2Train仿真进程上，等待其计算完成并接收到计算结果，将计算结果返回。

拿到两个算子的计算结果后，查看结果是否相符。

伪代码:

```python
torch_conv2d = torch.nn.Conv2d(1,1,3)
s2train_conv2d = dataflow.op.S2TrainConv2d(1,1,3)

weight = torch.rand((1,1,3,3))
torch_conv2d.weight = weight.copy()
s2train_conv2d.weight = weight.copy()


input = torch.rand((1,1,224,224))
torch_output = torch_conv2d(input)
s2train_output = s2train_conv2d(input)

assert torch_output==input_output
```

#### 整个网络的测试

##### step1 算子替换

将神经网络中的nn.Conv2d、nn.Linear等算子，转换为dataflow.op.S2TrainConv2d、dataflow.op.S2TrainLinear等算子

这样使得在计算过程中，卷积、全连接等计算可以放到S2Train仿真进程上来做

这一步可以用torch.fx的replace_pattern实现

##### step2 运行网络

替换完算子后，直接执行该网络。

各个算子前传、反传的调度，均由Pytorch的自动求导机制来完成。

对于卷积、全连接等S2Train实现了的算子，其前传和反传的计算会被放到S2Train仿真进程上完成计算；对于其他S2Train不支持的算子，则直接使用Pytorch完成计算。

伪代码如下：

```python
#输入
input = torch.rand((1,3,224,224))
s2train_input = input.copy()

#算子替换，生成s2train_model
s2train_model = op_replace(origin_model)

#测前传
out = origin_model(input)
s2train_out = s2train_model(s2train_input)
assert out==s2train_out

#测反传
s2train_loss = loss_function(out,label)
s2train_loss.backward()
loss = function(s2train_out,label)
loss.backward()
assert input.grad==s2train_input.grad



```

# simulator

模拟S2Train的接口，但其计算是使用软件方式调库直接实现的。在硬件那边可以仿真之前，用这个来测。

## 

