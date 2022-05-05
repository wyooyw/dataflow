import collections
class Net:
    def __init__(self):
        self.ops = set()
        self.first_op = None
    
    def add_operator(self,*ops):
        """添加算子到网络中
        """
        for op in ops:
            if self.first_op==None:
                self.first_op = op
            self.ops.add(op)
            op.net = self
    
    def remove_operator(self,*ops):
        """从网络中删除算子
        """
        for op in ops:
            self.ops.remove(op)
            op.net = None
    
    def __str__(self):
        """将网络转换为字符串
        """
        strs = []
        op = self.first_op
        while op:
            strs.append(str(op))
            op = list(op.successor)[0] if len(op.successor)>0 else None
        # strs = [str(op) for op in self.ops]
        return "\n".join(strs)

