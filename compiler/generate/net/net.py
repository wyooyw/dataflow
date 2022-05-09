import collections
from queue import Queue
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
        """将网络转换为字符串,按照算子的拓扑顺序
        """
        strs = []
        queue = Queue()
        queue.put(self.first_op)
        record = {}
        
        while not queue.empty():
            op = queue.get()
            if len(op.predecessor)>1:
                if op not in record:
                    record[op] = len(op.predecessor) - 1
                    continue
                elif record[op]>1:
                    record[op] -= 1
                    continue
            strs.append(str(op))
            for suc in op.successor:
                queue.put(suc)
        return "\n".join(strs)
