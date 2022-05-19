from bitarray import bitarray
class InstructionGenerator(object):
    def __init__(self,net):
        self.net = net
        self.instruction_list = []
        for op in net.topo():
            if hasattr(op,"to_instr"):
                self.add(op.to_instr())
        
    
    def add(self,instruction):
        self.instruction_list.append(instruction)
    
    def export(self):
        bits = bitarray()
        for instr in self.instruction_list:
            bits.extend(instr.export())
        return bits
    
    def export_to_file(self,path):
        bits = self.export()
        with open(path, 'wb') as f:
            bits.tofile(f)