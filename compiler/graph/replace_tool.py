from compiler.generate.net.net import Net
from compiler.generate.operator import Operator
from compiler.config.config import Config
from compiler.generate.op.conv import DualConv,ForwardConv,BackwardConv
from compiler.generate.op.softmax import DualSoftmax,ForwardSoftmax,BackwardSoftmax
from compiler.generate.op.entropy import DualEntropy,ForwardEntropy,BackwardEntropy
from compiler.generate.op.relu import DualRelu,ForwardRelu,BackwardRelu
from compiler.generate.op.flatten import DualFlatten,ForwardFlatten,BackwardFlatten
from compiler.generate.op.linear import DualLinear,ForwardLinear,BackwardLinear
from compiler.generate.op.add import DualAdd,ForwardAdd,BackwardAdd
from backends.sparse_train.op import *
from functools import reduce
class ReplaceTool:
    def __init__(self,net:Net,config_path:str):
        self.net = net
        self.config = Config().load(config_path)
        pass
    def replace_all(self):
        for item in self.config:
            patterns,replace = item["pattern"],item["replace"]
            pattern_class = [eval(pattern) for pattern in patterns]
            replace_class = eval(replace)
            self.replace(pattern_class,replace_class)

    def replace(self,pattern,replace):
        finder = Finder(net=self.net,pattern=pattern)
        find = finder.find()
        for f in find:
            replace_op = replace.replace_from(f)

            #合并后继节点
            origin = set(f)
            
            successors = reduce(lambda x,y:x|y, [op.successor for op in f]) - origin
            for successor in successors:
                successor.disconnect_predecessor(*origin)
                successor.connect_predecessor(replace_op)

            #合并前驱节点
            predecessors = reduce(lambda x,y:x|y, [op.predecessor for op in f]) - origin
            for predecessor in predecessors:
                predecessor.disconnect_successor(*origin)
                predecessor.connect_successor(replace_op)

            #替换网络中的算子
            net = f[0].net
            net.remove_operator(*origin)
            net.add_operator(replace_op)
            
            

class Finder:
    def __init__(self,net:Net,pattern:list[Operator]):
        self.net = net
        self.visit = set() #每个op是否已被访问
        self.pattern = pattern
        self.tmp = []
        self.tmp_idx = 0
        
    def find(self):
        result = []
        while True:
            begin_op = self.find_first_op()
            if begin_op==None:
                break
            op_seq = self.check(begin_op)
            if len(op_seq)>0:
                result.append(op_seq)
        return result

    def find_first_op(self):
        pattern_first = self.pattern[0]
        ret = None
        for op in self.net.ops:
            if type(op)==pattern_first and not op in self.visit:
                ret = op
                break
        return ret

    def check(self,begin_op:Operator):
        if type(begin_op) == self.pattern[0]:
            self.tmp = [begin_op]
            ret = self.dfs(begin_op,0)
        self.tmp = []
        self.visit.add(begin_op)

        if len(ret)==0:
            return ret

        #检查是否合法
        valid = reduce(lambda x,y:x&y, [len(r.successor)==1 for r in ret[0:-1]])
        if not valid:
            return []

        self.visit = self.visit | set(self.tmp)
        return ret

    def dfs(self,begin_op:Operator,step:int):
        ret = []
        if step==len(self.pattern)-1:
            # self.visit = self.visit | set(self.tmp)
            ret = [*self.tmp]
        else:
            successors = begin_op.successor
            for successor in successors:
                if not successor in self.visit:
                    if type(successor) == self.pattern[step+1]:
                        self.tmp.append(successor)
                        ret = self.dfs(successor,step+1)
                        self.tmp.pop()
                        if len(ret)>0:
                            break
        return ret

    