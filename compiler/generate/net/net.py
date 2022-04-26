import collections
class Net:
    def __init__(self):
        self.ops = set()
    
    def add_operator(self,*ops):
        for op in ops:
            self.ops.add(op)
            op.net = self
    
    def remove_operator(self,*ops):
        for op in ops:
            self.ops.remove(op)
            op.net = None
    
    def __str__(self):
        strs = [str(op) for op in self.ops]
        return "\n".join(strs)

