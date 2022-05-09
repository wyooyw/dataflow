import torch
import torch.nn as nn
from torch.nn import Module
import torch.fx as fx
from torch.fx import symbolic_trace,Graph,Node,replace_pattern
import operator
from compiler.config.config import Config

from compiler.generate.op.conv import DualConv
from compiler.generate.op.softmax import DualSoftmax
from compiler.generate.op.entropy import DualEntropy
from compiler.generate.op.relu import DualRelu
from compiler.generate.op.flatten import DualFlatten
from compiler.generate.op.linear import DualLinear
from compiler.generate.op.add import DualAdd
# from compiler.generate.op import *
from compiler.generate.net.net import Net

class Converter(object):
    """将pytorch模型转换为sparsetrain-IR
    """
    def __init__(self,module,in_shape=[32,3,32,32]):
        self.origin_module = module
        self.trace = symbolic_trace(self.origin_module)
        self.named_modules = dict(self.trace.named_modules())
        self.in_shape = in_shape
        self.net = Net()
        self.map = dict() #key:original ;value :(forward_new,backward_new)
        pass

    def get_in_shape(self,node):
        in_shape = []
        assert len(node.args)>0,"Length of node.arg should bigger than 0."
        arg = node.args[0]
        if arg.op=="placeholder":
            in_shape = self.in_shape
        if (arg.op=="call_module" or arg.op=="call_function") and arg in self.map:
            dual = self.map[arg]
            in_shape = dual.forward.out_shape
        assert len(in_shape)>0,f"Get in_shape error.node:{node}"
        return in_shape

    def convert(self):
        graph = self.trace.graph
        named_modules = self.named_modules
        config_call_module = Config().convert_config["call_module"]
        config_call_function = Config().convert_config["call_function"]
        for node in graph.nodes:
            if node.op=="placeholder":
                pass
            elif node.op=="output":
                assert len(node.args)==1,"Convert output error."
                in_shape = self.get_in_shape(node)
                dual_last = self.map[node.args[0]]
                dual_softmax = DualSoftmax.from_torch_module(in_shape=in_shape,module=None)
                dual_entropy = DualEntropy.from_torch_module(in_shape=in_shape,module=None)
                self.net.add_operator(dual_softmax.forward,dual_softmax.backward)
                self.net.add_operator(dual_entropy.forward,dual_entropy.backward)
                
                dual_last.forward.connect_successor(dual_softmax.forward)
                dual_softmax.forward.connect_successor(dual_entropy.forward)
                dual_last.backward.connect_predecessor(dual_softmax.backward)
                dual_softmax.backward.connect_predecessor(dual_entropy.backward)

                dual_entropy.forward.connect_successor(dual_entropy.backward)
            elif node.op=="call_module":
                # if node.target=="conv2":
                #     import ipdb
                #     ipdb.set_trace()
                flag = False
                for item in config_call_module:
                    module = named_modules[node.target]
                    if type(module) == eval(item["old"]):
                        in_shape = self.get_in_shape(node)
                        dual = eval(item["new"]).from_torch_module(in_shape=in_shape,module=module)
                        self.net.add_operator(dual.forward,dual.backward)
                        self.map[node] = dual
                        #加入前驱节点
                        for arg in node.args:
                            if arg in self.map:
                                predecessor_dual = self.map[arg]
                                predecessor_dual.forward.connect_successor(dual.forward)
                                predecessor_dual.backward.connect_predecessor(dual.backward)
                        flag = True
                        break
                assert flag,f"{type(module)} was not implemented in sparsetrain-IR."
            elif node.op=="call_function":
                flag = False
                for item in config_call_function:
                    if node.target==eval(item["old"]):
                        dual = eval(item["new"])(in_shape=in_shape)
                        self.net.add_operator(dual.forward,dual.backward)
                        self.map[node] = dual
                        #加入前驱节点
                        for arg in node.args:
                            if arg in self.map:
                                pre_dual = self.map[arg]
                                pre_dual.forward.connect_successor(dual.forward)
                                pre_dual.backward.connect_predecessor(dual.backward)
                    flag = True
                    break
                assert flag,f"{node.target} was not implemented in sparsetrain-IR."

# class MyNet(Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3,4,5)
#         self.relu1 = nn.ReLU()
#         # self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(4,5,5)
#         self.relu2 = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(512,256)
#         self.fc2 = nn.Linear(256,10)
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         # x = self.pool(x)
#         y = self.conv2(x)
#         y = self.relu2(y)
#         y = y + x
#         y = self.flatten(y)
#         y = self.fc1(y)
#         y = self.fc2(y)
#         return y

if __name__=="__main__":
    net = MyNet()
    converter = Converter(net,in_shape=[32,3,32,32])
    # print(converter.trace.graph)
    converter.convert()
    print(converter.net)

