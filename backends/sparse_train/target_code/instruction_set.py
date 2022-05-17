from bitarray import bitarray
class InstructionList(object):
    def __init__(self):
        self.instruction_list = []
    
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