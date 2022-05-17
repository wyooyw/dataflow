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
from compiler.generate.op.add import DualAdd,BackwardAdd
from compiler.generate.op.maxpool import DualMaxpool
from compiler.generate.op.batchnorm import DualBatchnorm
from compiler.generate.op.edge import DualEdge
from compiler.generate.op.split import DualSplit,ForwardSplit

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
                dual = DualEdge(in_shape=self.in_shape)
                self.net.add_operator(dual.forward,dual.backward)
                self.map[node] = dual
            elif node.op=="output":
                assert len(node.args)==1,"Convert output error."
                in_shape = self.get_in_shape(node)
                dual_last = self.map[node.args[0]]
                dual_softmax = DualSoftmax.from_torch_module(in_shape=in_shape,module=None)
                dual_entropy = DualEntropy.from_torch_module(in_shape=in_shape,module=None)
                self.net.add_operator(dual_softmax.forward,dual_softmax.backward)
                self.net.add_operator(dual_entropy.forward,dual_entropy.backward)
                
                dual_last.forward.connect_successor(dual_softmax.forward,_share_storage=True)
                dual_softmax.forward.connect_successor(dual_entropy.forward,_share_storage=True)
                dual_last.backward.connect_predecessor(dual_softmax.backward,_share_storage=True)
                dual_softmax.backward.connect_predecessor(dual_entropy.backward,_share_storage=True)

                dual_entropy.forward.connect_successor(dual_entropy.backward)
            elif node.op=="call_module":
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
                                predecessor_dual.forward.connect_successor(dual.forward,_share_storage=True)
                                predecessor_dual.backward.connect_predecessor(dual.backward,_share_storage=True)
                        if len(node.users)>1:
                            self._add_split(node,in_shape=in_shape)
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
                                pre_dual.forward.connect_successor(dual.forward,_share_storage=True)
                                pre_dual.backward.connect_predecessor(dual.backward,_share_storage=True)
                        if len(node.users)>1:
                            self._add_split(node,in_shape=in_shape)
                        flag = True
                        break
                assert flag,f"{node.target} was not implemented in sparsetrain-IR."
        self._clean_no_use()
    def _add_split(self,node,in_shape):
        assert len(node.users)==2,"Split num greater than 2,not implemented!"
        dual_split = DualSplit(in_shape)
        dual = self.map[node]
        dual.forward.connect_successor(dual_split.forward,_share_storage=True)
        dual.backward.connect_predecessor(dual_split.backward,_share_storage=True)
        self.net.add_operator(dual_split.forward,dual_split.backward)
        self.map[node] = dual_split

    def _clean_no_use(self):
        """移除ForwardSplit和BackwardAdd，这两个没用
        """
        for op in self.net.topo():
            if type(op)==BackwardAdd:
                print(f"[Conventer] Remove no-use node: {op.name}")
                predecessor_set = [*op.predecessor]
                successor_set = [*op.successor]
                for predecessor in predecessor_set:
                    predecessor.disconnect_successor(op)
                    predecessor.connect_successor(*successor_set)

                for successor in successor_set:
                    successor.disconnect_predecessor(op)
                    successor.connect_predecessor(*predecessor_set)
            elif type(op)==ForwardSplit:
                print(f"[Conventer] Remove no-use node: {op.name}")
                predecessor_set = [*op.predecessor]
                successor_set = [*op.successor]
                for predecessor in predecessor_set:
                    predecessor.disconnect_successor(op)
                    predecessor.connect_successor(*successor_set)

                for successor in successor_set:
                    successor.disconnect_predecessor(op)
                    successor.connect_predecessor(*predecessor_set)