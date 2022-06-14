from enum import Enum
import copy
from orderedset import OrderedSet
import collections
import copy


class Attrs:
    """维护算子中使用到的属性
    """
    def __init__(self):
        self.attrs = collections.OrderedDict()
        self.op = None
    
    def get(self,key):
        return self.attrs[key]

    def set(self,key,value):
        self.attrs[key] = value

    def __copy__(self):
        copy_self = type(self)()
        copy_self.attrs = copy.copy(self.attrs)
        copy_self.op = self.op
        return copy_self

def warn(cond,string):
    if not cond:
        print(string)
class Tensors:
    """维护算子中使用到的张量
    """
    def __init__(self):
        self.tensors = collections.OrderedDict()
        self.write_tensors = set()
        self.read_tensors = set()
        self.input = None
        self.output = None
        self.output_idx = -1
        self.input_idx = -1
        self.op = None #该Tensors的父元素
    
    def add_read_tensor(self,tensor_name):
        assert tensor_name in self.tensors
        warn(tensor_name not in self.read_tensors,f"Tensor {tensor_name} is already in read_tensors")
        warn(tensor_name not in self.write_tensors,f"Tensor {tensor_name} is already in write_tensors")
        
        self.read_tensors.add(tensor_name)
    
    def add_write_tensor(self,tensor_name):
        assert tensor_name in self.tensors
        warn(tensor_name not in self.read_tensors,f"Tensor {tensor_name} is already in read_tensors")
        warn(tensor_name not in self.write_tensors,f"Tensor {tensor_name} is already in write_tensors")
        self.write_tensors.add(tensor_name)

    def get_input(self):
        return [self.tensors[item] for item in self.input]

    def get_output(self):
        return [self.tensors[item] for item in self.output]

    def set_next_output(self,tensor):
        self.output_idx += 1
        assert self.output_idx < len(self.output),f"[{self.op.name}] output out of range,output_idx={self.output_idx},len(self.output)={len(self.output)}"
        self.tensors[self.output[self.output_idx]] = tensor

    def get_next_input(self):
        self.input_idx += 1
        assert self.input_idx < len(self.input),f"[{self.op.name}] input out of range,input_idx={self.input_idx},len(self.input)={len(self.input)}"
        return self.tensors[self.input[self.input_idx]]
    
    def get(self,key):
        return self.tensors[key]
    
    def set(self,key,value):
        self.tensors[key] = value

    def __copy__(self):
        """复制对象

        这里复制后,type(copy_self)会变成OpTensor,而不是子类,目前不影响,后续最好处理一下。
        """
        copy_self = Tensors()
        copy_self.tensors = copy.copy(self.tensors)
        copy_self.input = copy.copy(self.input)
        copy_self.output = copy.copy(self.output)
        copy_self.output_idx = self.output_idx
        copy_self.input_idx = self.input_idx
        copy_self.op = self.op
        return copy_self
        


class OperatorType(Enum):
    FORWARD = 1000
    BACKWARD = 1001
    WEIGHT_GRADIENT = 1002
    WEIGHT_UPDATE = 1003
    BACKEND = 1004
    
class Operator:
    def __init__(self,type:OperatorType,name="",attrs:Attrs=Attrs(),tensors:Tensors=Tensors(),in_shape=[],out_shape=[]):
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
            if op: #有时op会为None
                if _share_storage and op.get_tensors().output and self.get_tensors().input:
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
            if op: #有时op会为None
                if _share_storage and op.get_tensors().input and self.get_tensors().output:
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

    def remove_all_predecessor(self):
        """
        单向删除所有前驱节点
        """
        self.predecessor = OrderedSet()

    def remove_all_successor(self):
        """
        单向删除所有后继节点
        """
        self.successor = OrderedSet()

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

    def sim_run(self):
        pass
    
