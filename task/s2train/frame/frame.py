import numpy as np
from bitarray import bitarray
from bitarray.util import ba2hex
import math
from functools import reduce

def int_to_bits(num,width,name="Num",endian="big"):
    assert num>=0 and num<(1<<width),f"{name} should in [{0},{(1<<width)-1}],but got {num}"
    bits = bitarray(endian=endian)
    bits.frombytes(int(num).to_bytes(math.ceil(width/8),byteorder=endian, signed=False))
    if endian=="big":
        bits = bits[-width:]
    else:
        bits = bits[0:width]
    return bits

class LAD(object):
    """ LAD

    LAD means "Length-Address-Data",is the smallest unit in a Frame
    """
    def __init__(self,addr,data,):
        # make 'len'
        leng = reduce(lambda x,y:x*y,data.shape)

        # check
        self.check(leng,addr,data)

        # save attrs
        self.leng = leng
        self.addr = addr
        self.data = data

    def check(self,leng,addr,data):
        """ Check LAD's params

        'leng': int, [0,2^16)
        'addr': int, [0,2^32)
        'data': 
            1.tensor(half), numel(data)<2^14
            2.bitarray, len(data)<2^18
        """

        # check len 
        assert type(leng)==int, f"Type of 'leng' should be int, but got {type(leng)}"
        assert leng >= 0 and leng < (1<<16), f"Value of 'leng' should in [0,2^16), but got {leng}"

        # check addr
        assert type(addr)==int, f"Type of 'addr' should be int, but got {type(addr)}"
        assert addr >= 0 and leng < (1<<32), f"Value of 'addr' should in [0,2^32), but got {addr}"

        # check data
        if type(data)==np.ndarray:
            data = data.reshape(-1)
            assert data.dtype==np.half, f"Dtype of 'data' should be np.half, but got {data.dtype}"
            assert data.shape[0] < (1<<14), f"Numel of 'data' should in [0,2^14), but got {data.shape[0]}"
        elif type(data)==bitarray:
            assert len(data) < (1<18), f"Bit length of 'data' should in [0,2^18), but got {len(data)}"
        else:
            assert False,f"Type of 'data' should be np.ndarray or bitarray, but got {type(data)}"

    def to_str(self,use_hex=False):
        """ 以字符串的形式展示LAD结构的内容

        If use_hex==True, show number as hex type.Otherwise show origin data.
        """
        if use_hex==True:
            s = f"\tlen:{ba2hex(int_to_bits(self.leng,16))}\n"
            s += f"\taddr:{ba2hex(int_to_bits(self.addr,32))}\n"
            data_list = []
            for num in self.data:
                tmp = bitarray()
                tmp.frombytes(num.tobytes())
                data_list.append(ba2hex(tmp))
            s += f"\tdata:{','.join(data_list)}\n"
        else:
            s = f"\tlen:{self.leng}\n"
            s += f"\taddr:{hex(self.addr).upper()}\n"
            s += f"\tdata:{self.data}\n"

        return s

    @classmethod
    def zeros(self,shape):
        data = np.zeros(shape).astype(np.half)
        lad = LAD(data=data,addr=0)
        lad.leng = 0
        return lad
        

class Lane(object):
    """ Lane

    Lane is a list of LAD.
    """
    def __init__(self,LAD_list=None):
        # self.check(LAD_list)
        if LAD_list==None:
            self.LAD_list = []
        else:
            self.LAD_list =LAD_list
    
    def check(self,LAD_list):
        """ Check Lane

        'LAD_list' is a list of 'LAD'
        """
        assert type(LAD_list)==list
        if len(LAD_list)>0:
            for item in LAD_list:
                assert type(item)==LAD

    def append_lad(self,lad):
        assert type(lad)==LAD,f"Params of 'append_lad' should be LAD, but got {type(lad)}"
        self.LAD_list.append(lad)
    
    def __getitem__(self,key):
        assert type(key)==int
        assert key >= 0 and key < len(self.LAD_list)
        return self.LAD_list[key]
    
    def __len__(self):
        return len(self.LAD_list)


class Frame(object):
    def __init__(self,
                    lanes,
                    head=None,
                    transnum=None,
                    len=None,
                    reverse=None,
                    head_crc_64=None):
        # Check
        # self.check_lanes(lanes)

        self.head = head
        self.transnum = transnum
        self.len = len
        self.reverse = reverse
        self.head_crc_64 = head_crc_64
        self.lanes = lanes
        pass

    # def check_lanes(self,lanes):
    #     for four_lane in lanes:
    #         assert len(four_lane)==4
    #         for lane in four_lane:
    #             assert type(lane)==LAD
    
    def iter_lanes(self):
        for i in range(0,len(self.lanes[0])):
            yield self.lanes[0][i],self.lanes[1][i],self.lanes[2][i],self.lanes[3][i]

    def gather_len(self,LAD_0,LAD_1,LAD_2,LAD_3):
        """ Gather len from four LAD
        """
        gather_bits = bitarray()
        gather_bits.extend(int_to_bits(LAD_0.leng,width=16))
        gather_bits.extend(int_to_bits(LAD_1.leng,width=16))
        gather_bits.extend(int_to_bits(LAD_2.leng,width=16))
        gather_bits.extend(int_to_bits(LAD_3.leng,width=16))
        return gather_bits

    def gather_addr(self,LAD_0,LAD_1,LAD_2,LAD_3):
        """ Gather addr from four LAD

        1.gather high 16 bits of each LAD
        2.gather low 16 bits of each LAD
        """
        gather_bits = bitarray()
        gather_bits.extend(int_to_bits(LAD_0.addr>>16,width=16))
        gather_bits.extend(int_to_bits(LAD_1.addr>>16,width=16))
        gather_bits.extend(int_to_bits(LAD_2.addr>>16,width=16))
        gather_bits.extend(int_to_bits(LAD_3.addr>>16,width=16))
        mask = (1<<16)-1
        gather_bits.extend(int_to_bits(LAD_0.addr&mask,width=16))
        gather_bits.extend(int_to_bits(LAD_1.addr&mask,width=16))
        gather_bits.extend(int_to_bits(LAD_2.addr&mask,width=16))
        gather_bits.extend(int_to_bits(LAD_3.addr&mask,width=16))
        return gather_bits

    def gather_data(self,LAD_0,LAD_1,LAD_2,LAD_3):
        """ Gather data from four LAD
        """
        data_0 = LAD_0.data.reshape(1,-1)
        data_1 = LAD_1.data.reshape(1,-1)
        data_2 = LAD_2.data.reshape(1,-1)
        data_3 = LAD_3.data.reshape(1,-1)

        data = np.concatenate((data_0,data_1,data_2,data_3),axis=0)
        data = np.transpose(data,(1,0))
        data = data.astype(np.float16)

        bits = bitarray()
        bits.frombytes(data.tobytes())
        return bits


    def export(self,path):
        """ Export frame to file
        """
        bits = self.to_bits()
        with open(path,'wb') as f:
            bits.tofile(f)

    def to_bits(self):
        bits = bitarray()
        # LAD_i is in i'th lane
        for LAD_0,LAD_1,LAD_2,LAD_3 in self.iter_lanes():
            gather_len_bits = self.gather_len(LAD_0,LAD_1,LAD_2,LAD_3)
            gather_addr_bits = self.gather_addr(LAD_0,LAD_1,LAD_2,LAD_3)
            gather_data_bits = self.gather_data(LAD_0,LAD_1,LAD_2,LAD_3)
            bits.extend(gather_len_bits)
            bits.extend(gather_addr_bits)
            bits.extend(gather_data_bits)
        return bits
    
    def to_str(self,use_hex=False):
        s = ""
        index = 0
        for LAD_0,LAD_1,LAD_2,LAD_3 in self.iter_lanes():
            s += f"Lane0-LAD{index}:\n"
            s += LAD_0.to_str(use_hex)
            s += f"Lane1-LAD{index}:\n"
            s += LAD_1.to_str(use_hex)
            s += f"Lane2-LAD{index}:\n"
            s += LAD_2.to_str(use_hex)
            s += f"Lane3-LAD{index}:\n"
            s += LAD_3.to_str(use_hex)

            index += 1
        return s

if __name__=="__main__":
    lad0 = LAD(1,9,np.arange(0.0,16.0).astype(np.float16))
    lad1 = LAD(2,10,np.arange(16.0,32.0).astype(np.float16))
    lad2 = LAD(3,11,np.arange(32.0,48.0).astype(np.float16))
    lad3 = LAD(4,12,np.arange(48.0,64.0).astype(np.float16))
    lad4 = LAD(5,13,np.arange(64.0,80.0).astype(np.float16))
    lad5 = LAD(6,14,np.arange(80.0,96.0).astype(np.float16))
    lad6 = LAD(7,15,np.arange(96.0,112.0).astype(np.float16))
    lad7 = LAD(8,16,np.arange(112.0,128.0).astype(np.float16))
    lane0 = Lane([lad0,lad4])
    lane1 = Lane([lad1,lad5])
    lane2 = Lane([lad2,lad6])
    lane3 = Lane([lad3,lad7])
    frame = Frame([lane0,lane1,lane2,lane3])
    frame.export("test.bin")
    print(frame.to_str(use_hex=True))