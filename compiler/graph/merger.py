from compiler.generate.net.net import Net
from compiler.generate.operator import Operator
class Merger:
    def __init__(self,net:Net):
        self.net = net
        pass
    def merge(self):
        for pattern,replacement in config:
            find_ops = find(pattern)
            for ops in find_ops:
                get_class(replacement).merge(ops)

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
            if op_seq==None or len(op_seq)==0:
                break
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
        return ret

    def dfs(self,begin_op:Operator,step:int):
        ret = []
        if step==len(self.pattern)-1:
            self.visit = self.visit | set(self.tmp)
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

    