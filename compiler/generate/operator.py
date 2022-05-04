from enum import Enum
from compiler.target_gen.memory.mem_operator import MemOperator
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
import copy
class OperatorType(Enum):
    # 1~50 单个的算子
    FORWARD_CONV = 0
    FORWARD_RELU = 1
    FORWARD_MAXPOOL = 2

    # 51~100 融合算子
    FORWARD_FUSE_CONV_RELU = 51

    # 101~150 单个算子的反传
    BACKWARD_CONV = 100
    BACKWARD_RELU = 101
    BACKWARD_MAXPOOL = 102

    # 151~200 融合算子的反传
    BACKWARD_FUSE_CONV_RELU = 151

    # 201,202 分叉算子和聚合算子
    SPLIT = 201
    AGGREGATE = 202

class Operator:
    def __init__(self,type:OperatorType,name="",attrs:Attrs=Attrs(),tensors:OpTensors=OpTensors()):
        self.name = name        #算子名称
        self.type = type        #算子类型
        self.predecessor = set()   #前驱算子
        self.successor = set()     #后继算子
        self.net = None
        self.attrs = attrs      #算子属性
        attrs.op = self
        self.tensors = tensors  #算子使用的张量
        tensors.op = self
        
        # split节点有两个后继节点
        # aggregate算子有两个前驱节点
        # 其他算子只能由一个前驱和一个后继节点
        if self.type==OperatorType.SPLIT:
            self.max_predecessor = 1
            self.max_successor = 2
        elif self.type==OperatorType.AGGREGATE:
            self.max_predecessor = 2
            self.max_successor = 1
        else:
            self.max_predecessor = 1
            self.max_successor = 1
        
    def add_predecessor(self,*ops):
        """单向添加前驱算子
        """
        for op in ops:
            assert len(self.predecessor)<self.max_predecessor,f"[{self.name}] Can't add more predecessor."
            self.predecessor.add(op)

    def add_successor(self,*ops):
        """单向添加后继算子
        """
        for op in ops:
            assert len(self.successor)<self.max_successor,f"[{self.name}] Can't add more successor."
            self.successor.add(op)
    
    def connect_predecessor(self,*ops):
        """双向添加前驱算子
        """
        for op in ops:
            self.add_predecessor(op)
            op.add_successor(self)
    
    def connect_successor(self,*ops):
        """双向添加后继算子
        """
        for op in ops:
            self.add_successor(op)
            op.add_predecessor(self)
    
    def next_output_to(self,op):
        """将本算子的输出指向后继算子的输入
        """
        # TODO 改成算子申请其输出张量的内存，这样可以节省在spliiter的内存开销
        self.tensors.set_next_output(op.tensors.get_next_input())

    def next_input_from(self,op):
        """将本算子的输出指向后继算子的输入
        """
        # TODO 改成算子申请其输出张量的内存，这样可以节省在spliiter的内存开销
        self.tensors.set_next_input(op.tensors.get_output())  
    
    def remove_predecessor(self,*ops):
        """单向删除前驱节点
        """
        for op in ops:
            self.predecessor.discard(op)
    
    def remove_successor(self,*ops):
        """单向删除后继节点
        """
        for op in ops:
            self.successor.discard(op)

    def disconnect_predecessor(self,*ops):
        """双向删除前驱节点
        """
        for op in ops:
            self.remove_predecessor(op)
            op.remove_successor(self)

    def disconnect_successor(self,*ops):
        """双向删除后继节点
        """
        for op in ops:
            self.remove_successor(op)
            op.remove_successor(self)

    def __str__(self):
        """打印算子信息
        """
        input_names = [op.name for op in self.predecessor]
        output_names = [op.name for op in self.successor]
        return f"{self.name},(input={input_names}, output={output_names})"

    def __copy__(self):
        """复制算子

        对attrs和tensors进行复制
        tensors下的storage不复制,与原算子共享
        """
        copy_attrs = copy.copy(self.attrs)
        copy_tensors = copy.copy(self.tensors)
        #type(self)为子类
        copy_self = type(self)(attrs=copy_attrs,tensors=copy_tensors)
        return copy_self
