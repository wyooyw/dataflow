from bitarray import bitarray
import numpy as np
import math

GLOBAL_FORMAT_STRING_VERBOSE=True
GLOBAL_FORMAT_STRING_ORIGIN=True
GLOBAL_FORMAT_ENDIAN = "big"
ENDIAN_BIG = "big"
ENDIAN_LITTLE = "little"
class BitArrayFactory(object):

    @staticmethod
    def get(endian=GLOBAL_FORMAT_ENDIAN):
        bits = bitarray(endian=endian)
        return bits
    '''
    init_width以字节(8bit)为单位
    '''
    @staticmethod
    def getConst(endian=GLOBAL_FORMAT_ENDIAN,init_num=0,init_width=1):
        bits = BitArrayFactory.get(endian=endian)
        bits.frombytes(int(init_num).to_bytes(init_width,byteorder=GLOBAL_FORMAT_ENDIAN, signed=True))
        return bits

    '''
    init_width以bit为单位
    '''
    @staticmethod
    def bGetConst(endian=GLOBAL_FORMAT_ENDIAN,init_num=0,init_width=8):
        bits = BitArrayFactory.get(endian=endian)
        bits.frombytes(int(init_num).to_bytes(math.ceil(init_width/8),byteorder=GLOBAL_FORMAT_ENDIAN, signed=True))
        # bits = bits[-init_width:]
        if endian==ENDIAN_BIG:
            bits = bits[-init_width:]
        else:
            bits = bits[0:init_width]
        return bits

    

class Prefix(object):
    def __init__(self,core_idx=0,pe_array_idx=0,is_group=True):
        assert pe_array_idx>=0 and pe_array_idx<4
        assert core_idx>=0 and pe_array_idx<8
        self.core_idx = core_idx
        self.pe_array_idx = pe_array_idx
        self.is_group = is_group

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(BitArrayFactory.bGetConst(init_num=self.pe_array_idx,init_width=2))
        bits.extend(BitArrayFactory.bGetConst(init_num=self.core_idx,init_width=3))
        bits.extend([0]*2)
        bits.append(1 if self.is_group else 0)
        return bits

    def __str__(self):
        s = str(self.to_bits().to01())
        if GLOBAL_FORMAT_STRING_VERBOSE==True:
            s += f"({'G' if self.is_group else 'D'})"
        return s

class RawGroupNumber(object):
    def __init__(self,group_number):
        self.group_number = group_number

    def to_bits(self):
        bits = BitArrayFactory.getConst(init_num=self.group_number)
        return bits
    
    def __str__(self):
        s = str(self.to_bits().to01())
        if GLOBAL_FORMAT_STRING_VERBOSE==True:
            s += f"({self.group_number})"
        if GLOBAL_FORMAT_STRING_ORIGIN==True:
            s = f"{self.group_number}"
        return s

class GroupNumber(object):
    def __init__(self,group_number,prefix=Prefix(is_group=True)):
        self.prefix = prefix
        self.raw_group_number = RawGroupNumber(group_number)

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(self.prefix.to_bits())
        bits.extend(self.raw_group_number.to_bits())
        return bits
    
    def __str__(self):
        return str(self.prefix) + " | " + str(self.raw_group_number)

class RawDataCell(object):
    def __init__(self,data):
        assert type(data)==int or type(data)==float,data
        self.data = data

    def to_bits(self):
        return BitArrayFactory.getConst(init_num=self.data,init_width=2)

    def __str__(self):
        s = str(self.to_bits().to01())
        if GLOBAL_FORMAT_STRING_VERBOSE==True:
            s += f"({self.data})"
        if GLOBAL_FORMAT_STRING_ORIGIN==True:
            s = f"{self.data}"
        return s

class RawData(object):
    def __init__(self,data_list):
        if type(data_list)==np.ndarray:
            data_list = data_list.tolist()
        # assert type(data_list) == list
        if type(data_list[0]) == RawDataCell:
            self.data_list = data_list
        else:
            self.data_list = [RawDataCell(data) for data in data_list]
        # assert type(data_list[0]) == RawDataCell
        
    def to_bits(self):
        bits = BitArrayFactory.get()
        for data in self.data_list:
            bits.extend(data.to_bits())
        return bits

    def __str__(self):
        if GLOBAL_FORMAT_STRING_ORIGIN==True:
            return "\t".join([str(data)+" " for data in self.data_list])
        return " ".join([str(data) for data in self.data_list])

class Data(object):
    def __init__(self,data_list):
        self.prefix = Prefix(is_group=False)
        self.raw_data = RawData(data_list=data_list)

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(self.prefix.to_bits())
        bits.extend(self.raw_data.to_bits())
        return bits
    
    def __str__(self):
        return str(self.prefix) + " | " + str(self.raw_data)

class GroupData(object):
    def __init__(self,data_list=[]):
        assert type(data_list) == list
        if len(data_list)>0:
            assert type(data_list[0])==Data
        self.data_list = data_list
        

    def add(self,data):
        assert type(data)==Data
        self.data_list.append(data)

    def to_bits(self):
        bits = BitArrayFactory.get()
        for data in self.data_list:
            bits.extend(data.to_bits())
        return bits
    
    def __str__(self):
        return "\n".join([str(data) for data in self.data_list])

#Linear input

class LinearInputGroup(object):
    def __init__(self,group_number,group_data):
        assert type(group_number)==GroupNumber
        assert type(group_data)==GroupData
        self.group_number = group_number
        self.group_data = group_data

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(self.group_number.to_bits())
        bits.extend(self.group_data.to_bits())
        return bits

    def __str__(self):
        s = "group number:\n"
        s += str(self.group_number)
        s += "\ndata:\n"
        s += str(self.group_data)
        s += "\n"
        return s

class LinearInput(object):
    def __init__(self,linear_input_group_list=[]):
        self.linear_input_group_list = linear_input_group_list

    def add(self,linear_input_group):
        assert type(linear_input_group)==LinearInputGroup
        self.linear_input_group_list.append(linear_input_group)
    
    def to_bits(self):
        bits = BitArrayFactory.get()
        for group in self.linear_input_group_list:
            bits.extend(group.to_bits())
        return bits
    
    def __str__(self):
        s = ""
        for group in self.linear_input_group_list:
            s += str(group)
            s += "-"*10
            s += "\n"
        s = "="*10 + "\n" + s + "\n" + "="*10 
        return s

#Linear weight

class LinearWeightGroup(object):
    def __init__(self,group_number,group_data):
        assert type(group_number)==GroupNumber
        assert type(group_data)==GroupData
        self.group_number = group_number
        self.group_data = group_data

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(self.group_number.to_bits())
        bits.extend(self.group_data.to_bits())
        return bits

    def __str__(self):
        s = "group number:\n"
        s += str(self.group_number)
        s += "\ndata:\n"
        s += str(self.group_data)
        s += "\n"
        return s

class LinearWeight(object):
    def __init__(self,linear_weight_group_list=[]):
        self.linear_weight_group_list = linear_weight_group_list

    def add(self,linear_weight_group):
        assert type(linear_weight_group)==LinearWeightGroup
        self.linear_weight_group_list.append(linear_weight_group)
    
    def to_bits(self):
        bits = BitArrayFactory.get()
        for group in self.linear_weight_group_list:
            bits.extend(group.to_bits())
        return bits
    
    def __str__(self):
        s = ""
        for group in self.linear_weight_group_list:
            s += str(group)
            s += "-"*10
            s += "\n"
        s = "="*10 + "\n" + s + "\n" + "="*10 
        return s
    

#ConvInput
class ConvInputGroup(object):
    def __init__(self,group_number,group_data):
        assert type(group_number)==GroupNumber
        assert type(group_data)==GroupData
        self.group_number = group_number
        self.group_data = group_data

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(self.group_number.to_bits())
        bits.extend(self.group_data.to_bits())
        return bits

    def __str__(self):
        s = "group number:\n"
        s += str(self.group_number)
        s += "\ndata:\n"
        s += str(self.group_data)
        s += "\n"
        return s

class ConvInput(object):
    def __init__(self,conv_input_group_list=[]):
        self.conv_input_group_list = conv_input_group_list

    def add(self,conv_input_group):
        assert type(conv_input_group)==ConvInputGroup
        self.conv_input_group_list.append(conv_input_group)
    
    def to_bits(self):
        bits = BitArrayFactory.get()
        for group in self.conv_input_group_list:
            bits.extend(group.to_bits())
        return bits
    
    def __str__(self):
        s = ""
        for group in self.conv_input_group_list:
            s += str(group)
            s += "-"*10
            s += "\n"
        s = "="*10 + "\n" + s + "\n" + "="*10 
        return s

#ConvWeight
class ConvWeightGroup(object):
    def __init__(self,group_number,group_data):
        assert type(group_number)==GroupNumber
        assert type(group_data)==GroupData
        self.group_number = group_number
        self.group_data = group_data

    def to_bits(self):
        bits = BitArrayFactory.get()
        bits.extend(self.group_number.to_bits())
        bits.extend(self.group_data.to_bits())
        return bits

    def __str__(self):
        s = "group number:\n"
        s += str(self.group_number)
        s += "\ndata:\n"
        s += str(self.group_data)
        s += "\n"
        return s

class ConvWeight(object):
    def __init__(self,conv_weight_group_list=[]):
        self.conv_weight_group_list = conv_weight_group_list

    def add(self,conv_weight_group):
        assert type(conv_weight_group)==ConvWeightGroup
        self.conv_weight_group_list.append(conv_weight_group)
    
    def to_bits(self):
        bits = BitArrayFactory.get()
        for group in self.conv_weight_group_list:
            bits.extend(group.to_bits())
        return bits
    
    def __str__(self):
        s = ""
        for group in self.conv_weight_group_list:
            s += str(group)
            s += "-"*10
            s += "\n"
        s = "="*10 + "\n" + s + "\n" + "="*10 
        return s


if __name__=="__main__":
    print(BitArrayFactory.bGetConst(endian=ENDIAN_BIG,init_num=6,init_width=5))