from enum import Enum
from compiler.target_gen.memory.mem_operator import MemOperator
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
import copy
from orderedset import OrderedSet
class OperatorType(Enum):
    # 1~50 单个的算子
    FORWARD_CONV = 0
    FORWARD_RELU = 1
    FORWARD_MAXPOOL = 2
    FORWARD_FLATTEN = 3
    FORWARD_SOFTMAX = 4
    FORWARD_ENTROPY = 5
    FORWARD_LINEAR = 6

    # 51~100 融合算子
    FORWARD_CONV_RELU = 51

    # 101~150 单个算子的反传
    BACKWARD_CONV = 100
    BACKWARD_RELU = 101
    BACKWARD_MAXPOOL = 102
    BACKWARD_FLATTEN = 103
    BACKWARD_SOFTMAX = 104
    BACKWARD_ENTROPY = 105
    BACKWARD_LINEAR = 106

    # 151~200 融合算子的反传
    BACKWARD_CONV_RELU = 151

    # 201,202 分叉算子和聚合算子
    SPLIT = 201
    AGGREGATE = 202

    FORWARD = 1000
    BACKWARD = 1001
    BACKEND = 1002
    

class Operator:
    def __init__(self,type:OperatorType,name="",attrs:Attrs=Attrs(),tensors:OpTensors=OpTensors(),in_shape=[],out_shape=[]):
        self.name = name        #算子名称
        self.type = type        #算子类型
        self.predecessor = OrderedSet()#set()   #前驱算子
        self.successor = OrderedSet()#set()     #后继算子
        self.net = None
        self.attrs = attrs      #算子属性
        attrs.op = self
        self.tensors = tensors  #算子使用的张量
        tensors.op = self
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # split节点有两个后继节点
        # aggregate算子有两个前驱节点
        # 其他算子只能由一个前驱和一个后继节点
        # if self.type==OperatorType.SPLIT:
        #     self.max_predecessor = 1
        #     self.max_successor = 2
        # elif self.type==OperatorType.AGGREGATE:
        #     self.max_predecessor = 2
        #     self.max_successor = 1
        # else:
        #     self.max_predecessor = 1
        #     self.max_successor = 1
        self.max_predecessor = 10
        self.max_successor = 10
        
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
    
    def connect_predecessor(self,*ops,_share_storage=False):
        """双向添加前驱算子
        """
        for op in ops:
            if _share_storage:
                if type(self.get_tensors().input)==list:
                    op.get_tensors().output.storage.same_as(self.get_tensors().input[len(self.predecessor)].storage)
                else:
                    op.get_tensors().output.storage.same_as(self.get_tensors().input.storage)
            self.add_predecessor(op)
            op.add_successor(self)
    
    def connect_successor(self,*ops,_share_storage=False):
        """双向添加后继算子
        """
        for op in ops:
            if _share_storage:
                if type(op.get_tensors().input)==list:
                    op.get_tensors().input[len(op.predecessor)].storage.same_as(self.get_tensors().output.storage)
                else:
                    op.get_tensors().input.storage.same_as(self.get_tensors().output.storage)
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
            # print(f"[{self.name}] remove predecessor {op.name}")
            self.predecessor.discard(op)
    
    def remove_successor(self,*ops):
        """单向删除后继节点
        """
        for op in ops:
            # print(f"[{self.name}] remove successor {op.name}")
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
            op.remove_predecessor(self)

    def __str__(self):
        """打印算子信息
        """
        input_names = [op.name for op in self.predecessor]
        output_names = [op.name for op in self.successor]
        is_backend = '*' if self.type==OperatorType.BACKEND else ''
        return f"{is_backend}{self.name},(predecessor={input_names}, successor={output_names}),in_shape={self.in_shape},out_shape={self.out_shape}"

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

    @classmethod
    def get_out_shape_by_in_shape(cls,in_shape,attr):
        """根据输入张量shape和算子的属性,推算输出张量shape
        """
        assert False,"This function should be implemented."

    @classmethod
    def get_in_shape_by_out_shape(cls,out_shape,attr):
        """根据输出张量shape和算子的属性,推算输入张量shape
        """
        assert False,"This function should be implemented."

    def get_tensors(self):
        """获取张量所属的所有tensor
        """
        return self.tensors
    