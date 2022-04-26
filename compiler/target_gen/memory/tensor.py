from compiler.target_gen.memory.storage import Storage
class Tensor:
    count = 0
    def __init__(self,storage:Storage,shape:tuple[int],addr:int=-1):
        self.addr = addr
        self.storage = storage
        self.shape = shape
        self.ndim = len(shape)
        self.id = Tensor.count
        Tensor.count += 1

    def __str__(self):
        return f"[tensor] id={self.id} shape={self.shape}"
    
    def export(self):
        return [self.ndim,self.shape,self.storage.addr]
