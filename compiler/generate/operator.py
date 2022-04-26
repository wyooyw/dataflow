from enum import Enum
from compiler.target_gen.memory.mem_operator import MemOperator
from compiler.generate.op.attrs.attrs import Attrs
from compiler.generate.op.tensors.op_tensors import OpTensors
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
        self.predecessor = []   #前驱算子
        self.successor = []     #后继算子
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
        
    def add_predecessor(self,op,need_another_connect=True):
        """添加前驱算子
        """
        assert len(self.predecessor)<self.max_predecessor,f"[{self.name}] Can't add more predecessor."
        if need_another_connect:
            op.add_successor(self,need_another_connect=False)
        self.predecessor.append(op)

    def add_successor(self,op,need_another_connect=True):
        """添加后继算子
        """
        assert len(self.successor)<self.max_successor,f"[{self.name}] Can't add more successor."
        
        if need_another_connect:
            op.add_predecessor(self,need_another_connect=False)
        self.successor.append(op)
        #TODO: 将“计算顺序”与“输入输出张量关联”解耦
        #在第二阶段 算子融合时，只改变计算顺序，不改变输入输出的张量关联
        self.tensors.set_next_output(op.tensors.get_next_input()) # TODO 改成算子申请其输出张量的内存，这样可以节省在spliiter的内存开销

    def __str__(self):
        """打印算子信息
        """
        input_names = [op.name for op in self.predecessor]
        output_names = [op.name for op in self.successor]
        return f"{self.name},(input={input_names}, output={output_names})"
