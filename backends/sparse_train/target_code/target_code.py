from bitarray import bitarray
class TargetCodeSet(object):
    def __init__(self):
        pass

class TargetCode(object):
    def __init__(self):
        pass

class TargetCodeSend(TargetCode):
    def __init__(self,tensor,port):
        super().__init__()
        self.tensor = tensor
        self.port = port
    
    def generate(self):
        instruction = Instruction(num1=tensor.addr,num2=port.port)
        return instruction

    def __str__(self):
        s = "send {tensor} to {port}".format(tensor=self.tensor.name,port=self.port.name)
        return s

class TargetCodeRecive(TargetCode):
    def __init__(self,tensor,port):
        super().__init__()
        self.tensor = tensor
        self.port = port