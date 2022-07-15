from frame import *
import numpy as np
import sys
# import compiler.utils.utils as padding_inside
class FrameMaker(object):
    """ Convert tensor to frame
    """
    def __init__(self):
        pass
class FrameFactory(object):
    """ 算子转换为frame
    """
    def __init__(self,tile_len=4):
        self.tile_len = tile_len

        # Address of FM group 0~3
        self.ADDR_FM_GROUP = [0xB000_0000,
                            0xB001_0000,
                            0xB002_0000,
                            0xB003_0000]
        self.ADDR_WM = [
            # core 0
            0X4080_0000,
            0X4090_0000,
            0X40A0_0000,
            0X40B0_0000,
            0X40C0_0000,
            0X40D0_0000,
            0X40E0_0000,
            0X40F0_0000,
            # core 1
            0X4180_0000,
            0X4190_0000,
            0X41A0_0000,
            0X41B0_0000,
            0X41C0_0000,
            0X41D0_0000,
            0X41E0_0000,
            0X41F0_0000,
            # core 2...
        ]

        # Size of WM
        WM_SIZE = 8*1024 #B

        # Number of WM
        WM_NUM = 64
        pass

    def gen_mem_frames(self,op):
        """ Generate frames of WM and FM of op.
        """
        if type(op)==ForwardConv:
            # Forward Conv
            tensor_in_fm = op.tensors.get_data("input")
            tensor_in_wm = op.tensors.get_data("weight")
            stride = op.attrs.get("stride")
            kernel_size = op.attrs.get("kernel_size")
            assert (stride==1 or stride==2) and (kernel_size==1 or kernel_size==3)
            if stride==1:
                fm = self.convert_conv_forward_fm(tensor_in_fm)
                wm = self.convert_conv_forward_wm(tensor_in_wm)
            elif stride==2 and kernel_size==1:
                fm = self.convert_conv_forward_stride_2_kernel_1_fm(tensor_in_fm)
                wm = self.convert_conv_forward_wm(tensor_in_wm)
            elif stride==2 and kernel_size==3:
                fm = self.convert_conv_forward_stride_2_kernel_3_fm(tensor_in_fm)
                wm = self.convert_conv_forward_stride_2_kernel_3_wm(tensor_in_wm)
        
        elif type(op)==BackwardConv:
            # Backward Conv
            tensor_in_fm = op.tensors.get_data("output_grad")
            tensor_in_wm = op.tensors.get_data("weight")
            assert (stride==1 or stride==2) and (kernel_size==1 or kernel_size==3)
            if stride==1 or (stride==2 and kernel_size==1):
                fm = self.convert_conv_backward_fm(tensor_in_fm)
                wm = self.convert_conv_backward_wm(tensor_in_wm)
            elif stride==2 and kernel_size==3:
                fm = self.convert_conv_backward_stride_2_kernel_3_fm(tensor_in_fm)
                wm = self.convert_conv_backward_wm(tensor_in_wm)

        elif type(op)==WGConv:
            # Weight Gradient Conv
            tensor_in_fm = op.tensors.get_data("output_grad")
            tensor_in_wm = op.tensors.get_data("input")
            assert (stride==1 or stride==2) and (kernel_size==1 or kernel_size==3)
            if stride==1 or (stride==2 and kernel_size==1):
                fm = self.convert_conv_wg_fm(tensor_in_fm)
                wm = self.convert_conv_wg_wm(tensor_in_wm)
            elif stride==2 and kernel_size==3:
                fm = self.convert_conv_wg_stride_2_kernel_3_fm(tensor_in_fm)
                wm = self.convert_conv_wg_wm(tensor_in_wm)

        elif type(op)==ForwardLinear:
            # Forward Linear
            tensor_in_fm = op.tensors.get_data("input")
            tensor_in_wm = op.tensors.get_data("weight")
            fm = self.convert_linear_forward_fm(tensor_in_fm)
            wm = self.convert_linear_forward_wm(tensor_in_wm)

        elif type(op)==BackwardLinear:
            # Backward Linear
            tensor_in_fm = op.tensors.get_data("output_grad")
            tensor_in_wm = op.tensors.get_data("weight")
            fm = self.convert_linear_backward_fm(tensor_in_fm)
            wm = self.convert_linear_backward_wm(tensor_in_wm)

        elif type(op)==WGLinear:
            # Weight Gradient Linear
            tensor_in_fm = op.tensors.get_data("output_grad")
            tensor_in_wm = op.tensors.get_data("input")
            fm = self.convert_linear_wg_fm(tensor_in_fm)
            wm = self.convert_linear_wg_wm(tensor_in_wm)

        else:
            assert False,f"Type of 'op' should be [Forward|Backward|WG][Conv|Linear], but got {type(op)}"
        
        return fm, wm

    def build_frame(self, tensor, axis_type, axis_tail_hook):
        """ Build frames from a tensor.

        Params:
            tensor: np.ndarray
            axis_type: 
                A list, the length of it equals tensor.ndim.
                Each element of 'axis_type' is in [0,1,2,3]:
                    0 means the axis corresponding one or several frame.
                    1 means the axis corresponding lad row.
                    2 means the axis corresponding four lad in one lad row.
                    3 means the axis corresponding the dimension in a lad's data.
            axis_tail_hook:
                A list, the length of it equals tensor.ndim.
                Each element of 'axis_tail_hook' is None or function:
        """
        ndim = tensor.ndim
        assert len(axis_type)==ndim
        # assert len(axis_tail_hook)==ndim
        
        # Check 'axis_type' use a state machine
        state = 0 # 0:frame; 1:lad_height; 2:lad_width; 3: data
        for item in axis_type:
            if state==0:
                if item==0: continue
                elif item==1: state = 1
                else: assert False
            elif state==1:
                if item==1: continue
                elif item==2: state = 2
                else: assert False
            elif state==2:
                if item==3: state = 3
                else: assert False
            elif state==3:
                if item==3: continue
                else: assert False
            else: assert False

        # Manage axis_tail_hook
        axis_tail_hook_dict = {}
        for cond,func in axis_tail_hook:
            key = max(cond)
            ls = [(i not in cond) for i in range(ndim)]
            axis_tail_hook_dict[key] = (ls,func)

        def loop_nest(data,step,is_tail_list):
            """
            """
            if axis_type[step]==0:
                frame_list = []
                for index, sub_data in enumerate(data):
                    if index==len(data)-1:
                        is_tail_list[step] = True
                        if step in axis_tail_hook_dict:
                            cond_tuple,func = axis_tail_hook_dict[step]
                            valid = reduce([x or y for x,y in zip(is_tail_list,cond_tuple)],
                                    lambda x,y: x and y)
                            if valid: 
                                sub_data = func(sub_data)
                    if axis_type[step+1]==0:
                        frames = loop_nest(sub_data,step+1,is_tail_list)
                    else:
                        lad_row_list = loop_nest(sub_data,step+1,is_tail_list)
                        lanes = [Lane(),Lane(),Lane(),Lane()]
                        for lad_row in lad_row_list:
                            for idx in range(4):
                                lanes[idx].append_lad(lad_row[idx])
                        frame = Frame(lanes=lanes)
                        frames = [frame]
                    frame_list.extend(frames)
                is_tail_list[step] = False
                return frame_list
            elif axis_type[step]==1:
                lad_list = []
                for index, sub_data in enumerate(data):
                    if index==len(data)-1:
                        is_tail_list[step] = True
                        if step in axis_tail_hook_dict:
                            cond_tuple,func = axis_tail_hook_dict[step]
                            valid = reduce(lambda x,y: x and y,
                                [x or y for x,y in zip(is_tail_list,cond_tuple)])
                            if valid: 
                                sub_data = func(sub_data) 
                    lads = loop_nest(sub_data, step+1, is_tail_list)
                    lad_list.extend(lads)
                is_tail_list[step] = False
                return lad_list
            elif axis_type[step]==2:
                lad_row = []
                for index, sub_data in enumerate(data):
                    if index==len(data)-1:
                        is_tail_list[step] = True
                        if step in axis_tail_hook_dict:
                            cond_tuple,func = axis_tail_hook_dict[step]
                            valid = reduce([x or y for x,y in zip(is_tail_list,cond_tuple)],
                                    lambda x,y: x and y)
                            if valid: 
                                sub_data = func(sub_data)
                    tensor = loop_nest(sub_data,step+1,is_tail_list)
                    lad = LAD(data=tensor, addr=self.ADDR_FM_GROUP[index])
                    lad_row.append(lad)
                for i in range(4-len(lad_row)):
                    lad_row.append(LAD.zeros(shape=sub_data.shape))
                is_tail_list[step] = False
                return [lad_row]
            elif axis_type[step]==3:
                new_tensors = []
                if step == len(axis_type) - 1:
                    return data.reshape(-1)
                for index, sub_data in enumerate(data):
                    if index==len(data)-1:
                        if step in axis_tail_hook_dict:
                            cond_tuple,func = axis_tail_hook_dict[step]
                            valid = reduce([x or y for x,y in zip(is_tail_list,cond_tuple)],
                                    lambda x,y: x and y)
                            if valid: 
                                sub_data = func(sub_data)
                        is_tail_list[step] = True
                    sub_tensor = loop_nest(sub_data, step+1, is_tail_list)
                    new_tensors.append(sub_tensor)
                new_tensor = np.concatenate(new_tensors,axis=0)
                return new_tensor

        frames = loop_nest(tensor,0,[False]*ndim)
        return frames

    def pad(self,tensor,dim,window,overlap=0):
        ndim = tensor.ndim
        length = tensor.shape[dim]
        if length <= window:
            mod = length
            overlap = 0
        else:
            mod = (length - window) % (window - overlap)
        pad = 0
        if mod>0:
            pad = window - overlap - mod
            pad_list = [(0,0)]*ndim
            pad_list[dim] = (0,pad)
            tensor = np.pad(tensor,pad_list)
            # tensor = np.pad(tensor,((0,0),(0,0),(0,0),(0,pad)))
        return tensor,pad
    
    def convert_linear_forward_wm(self,tensor):
        def _convert_linear_forward_weight(tensor):
            # 1. Check params
            out_features, in_features = tensor.shape
            assert out_features%(16*4) == 0
            assert out_features // 16 <= WM_NUM
            assert in_features * 16 * 2 <= WM_SIZE

            # 2. Reshape tensors
            tensor = tensor.reshape(out_features//16//4,4,16,in_features)
            tensor = np.transpose(tensor,(0,1,3,2))

            # 3. Make frame
            pe_num = out_features//16//4
            lanes = [Lane(),Lane(),Lane(),Lane()]
            for b in range(pe_num):
                for lane_idx in range(4):
                    col_idx = b*4+lane_idx
                    lad = LAD(data=tensor[p][lane_idx],addr=self.ADDR_WM_GROUP[col_idx])
                    lanes[lane_idx].append_lad(lad)
            frame = Frame(lanes=lanes)
            return frame

        # Split the big tensor
        out_features, in_features = tensor.shape
        I = in_features
        K = out_features // 4 // 16
        I_prime = WM_SIZE // 2 // 16
        K_prime = WM_NUM // 4
        frames = []
        for i in range(0,I,I_prime):
            for k in range(0,K,K_prime):
                small_tensor = tensor[k:k+K_prime,i:i+I_prime]
                frame = _convert_linear_forward_weight(small_tensor)
                franes.append(frame)
        return frames
    
        
    def convert_linear_forward_fm(self,tensor):
        # 1. Check params
        assert tensor.ndim == 2
        # assert tensor.shape[0] % 4 == 0

        # 2. Pad at batch dimension
        tensor,pad_height = self.pad(tensor,dim=0,window=4,overlap=0)

        # 3. Reshape tensor
        batch,in_features = tensor.shape
        tensor = tensor.reshape(1,batch//4, 4, in_features)

        # 4.Build frame
        frame_num,lad_height,lad_width,in_features = tensor.shape
        axis_type = [0,1,2,3]
        axis_tail_hook = [[(1,),lambda t:t[:lad_width-pad_height,:]]]
        frames = self.build_frame(tensor,axis_type,axis_tail_hook)

        return frames

    def convert_linear_backward_fm(self,tensor):
        """ Convert 'output gradient' into frame when doing back prop in linear layer.

        It do the same thing as 'convert_linear_forward_fm()'

        Params:
            tensor: Output gradient.
                    It will be put at feature memory.
        Returns:
            frames
        """
        return convert_linear_forward_fm(tensor)

    def convert_linear_backward_wm(self,tensor):
        """ Convert 'weight' into frame when doing back prop in linear layer.

        It almost do the same thing as 'convert_linear_forward_wm()'

        Params:
            tensor: Output gradient.
                    It will be put at weight memory.
        Returns:
            frame
        """
        tensor = np.transpose(tensor,(1,0))
        return convert_linear_forward_wm(tensor)


    def convert_linear_wg_wm(self,tensor):
        """ Convert 'input feature' into frame when doing weight-gradient in linear layer.

        Params:
            tensor: input feature.
                    It will be put at weight memory.
        Returns:
            frame
        """
        assert False,"Not implemented yet."
    
    def convert_linear_wg_fm(self,tensor):
        """ Convert 'output gradient' into frame when doing weight-gradient in linear layer.

        Params:
            tensor: output gradient.
                    It will be put at feature memory.
        Returns:
            frame
        """
        assert False,"Not implemented yet."


    def convert_conv_forward_wm(self,tensor):
        """ Convert 'weight' of a conv layer into frame.

        Params:
            tenor: weight
        Output:
            frame: a 'Frame' object
        """

        # 1. Check params
        assert type(tensor)==np.ndarray
        assert tensor.ndim==4

        # 2. Reshape weight
        batch,channel,height,width = tensor.shape
        tensor = tensor.reshape(batch//4,4,channel,height,width)
        tensor = np.transpose(tensor,(0,1,3,2,4))
        tensor = tensor.reshape(batch//4,4,height*channel*width)

        # 3. Make frame
        batch,four,group_length = tensor.shape
        lanes = [Lane(),Lane(),Lane(),Lane()]
        for b in range(batch):
            for lane_idx in range(4):
                col_idx = b*4+lane_idx
                lad = LAD(data=tensor[b][lane_idx],addr=self.ADDR_WM[col_idx])
                lanes[lane_idx].append_lad(lad)

        frame = Frame(lanes=lanes)
        print(frame.to_str())

    def convert_conv_forward_fm(self,tensor,kernel_size=3,padding=0):
        """ Convert 'input feature' of a conv layer into frame.

        The conv op must satisfied: stride==1
        A tile is a (channel,tile_width) tensor
        The 0st tile goes to fm0, 
        The 1st tile goes to fm1,and so on.
        
        Params:
            tensor: input feature map
            padding: zeros out side input fmap
        
        Returns:
            frame: a 'Frame' object
        """

        # 1. Check params
        assert type(tensor)==np.ndarray
        assert tensor.ndim==4
        assert type(padding)==int and padding>=0 and padding<=2
        assert type(kernel_size)==int and kernel_size>=1
        
        # 2. Padding outside
        tensor = np.pad(tensor,((0,0),(0,0),(padding,padding),(padding,padding)))

        # 3. Padding width and height. Make the following steps easier.
        # Paddings will be removed at the final frame-build step.
        overlap = kernel_size - 1
        tensor,pad_width = self.pad(tensor,dim=3,window=self.tile_len,overlap=overlap)
        tensor,pad_height = self.pad(tensor,dim=2,window=4,overlap=0)

        # 4.Fit the overlap
        batch,channel,height,width = tensor.shape
        new_tensors = []
        step = self.tile_len-overlap
        for i in range(0,width-overlap,step):
            tmp = tensor[:,:,:,i:i+step+overlap]
            tmp = tmp.reshape(1,*tmp.shape)
            new_tensors.append(tmp)
        tensor = np.concatenate(new_tensors,axis=0)
        tensor = np.transpose(tensor,(1,0,2,3,4))

        # 5.Merge 'tile_width' and 'channel' axis to a 'group' axis.
        batch,tile_column,channel,height,tile_width = tensor.shape
        tensor = np.transpose(tensor,(0,1,3,2,4))
        tensor = tensor.reshape(batch,tile_column,height//4,4,channel,tile_width)

        # 6.Build frame
        batch,tile_column,lad_height,lad_width,channel,tile_width = tensor.shape
        axis_type = [0,1,1,2,3,3]
        axis_tail_hook = [
                        [(1,), lambda t:t[:,:,:,:tile_width-pad_width] ], #切掉宽度上多余的padding
                        [(2,), lambda t:t[:lad_width-pad_height,:,:] ], #切掉高度上多余的padding
                        ]
        frames = self.build_frame(tensor,axis_type,axis_tail_hook)
        
        return frames
        

    def convert_conv_forward_stride_2_kernel_3_fm(self,tensor,padding=0):
        """ Convert 'input feature' of a conv layer into frame.

        The conv op must satisfied: stride==2, kernel_size==3.
        Different from 'convert_conv_forward_fm':
            the 0st and 1st tile is give to fm0, 
            the 2st and 3st tile is give to fm1, and so on.
        
        Params:
            tensor: input feature map
            padding: zeros out side input fmap
        
        Returns:
            frame: a 'Frame' object
        """
        # 1. Check params
        assert type(tensor)==np.ndarray
        assert tensor.ndim==4
        assert type(padding)==int and padding>=0 and padding<=2
        assert tensor.shape[2]%2==0 ,"Height must be a even number, otherwise the dataflow will confuse."
        
        # 2. Padding outside
        tensor = np.pad(tensor,((0,0),(0,0),(padding,padding),(padding,padding)))

        # 3. Padding width and height.
        # If tile_len is a even number, overlap should be 2. Otherwise, set overlap=1 is enough.
        overlap = 2 if self.tile_len%2==0 else 1
        tensor,pad_width = self.pad(tensor,dim=3,window=self.tile_len,overlap=overlap)
        tensor,pad_height = self.pad(tensor,dim=2,window=8,overlap=0)

        # 4.Fit the overlap
        batch,channel,height,width = tensor.shape
        new_tensors = []
        step = self.tile_len-overlap
        for i in range(0,width-overlap,step):
            tmp = tensor[:,:,:,i:i+step+overlap]
            tmp = tmp.reshape(1,*tmp.shape)
            new_tensors.append(tmp)
        tensor = np.concatenate(new_tensors,axis=0)
        tensor = np.transpose(tensor,(1,0,2,3,4))

        # 5.Merge 'tile_width' and 'channel' axis to a 'group' axis.
        batch,tile_column,channel,height,tile_width = tensor.shape
        tensor = tensor.reshape(batch,tile_column,channel,height//2,2,tile_width)
        tensor = np.transpose(tensor,(0,1,3,4,2,5))
        tensor = tensor.reshape(batch,tile_column,height//8,4,2,channel,tile_width)

        # 6.Build frames
        batch,tile_column,lad_height,lad_width,two,channel,tile_width = tensor.shape
        axis_type = [0,1,1,2,3,3,3]
        axis_tail_hook = [
            [(1,), lambda t:t[:,:,:,:,:tile_width-pad_width] ],
            [(2,), lambda t:t[:4-pad_height//2,:,:,:] ],
        ]
        frames = self.build_frame(tensor,axis_type,axis_tail_hook)

        return frames
        
    def convert_conv_forward_stride_2_kernel_1_fm(self,tensor,padding=0):
        """ Convert 'input feature' of a conv layer into frame.

        The conv op must satisfied: stride==2, kernel_size==1.
        Different from 'convert_conv_forward_fm':
            Downsample the input feature first.
        
        Params:
            tensor: input feature map
            padding: zeros out side input fmap
        
        Returns:
            frames: list of 'Frame' object
        """
        # 1. Check params
        assert type(tensor)==np.ndarray
        assert tensor.ndim==4
        assert type(padding)==int and padding>=0 and padding<=2
        
        # 2. Padding outside
        tensor = np.pad(tensor,((0,0),(0,0),(padding,padding),(padding,padding)))

        # 3. Downsample
        tensor = tensor[:,:,::2,::2]

        # 4. Deal as a stride-1 conv
        frames = self.convert_conv_forward_fm(tensor,kernel_size=1,padding=0)
        
        return frames

    def convert_conv_backward_fm(self,tensor,kernel_size=3):
        """ Convert 'output gradient' of a conv layer into frame.

        The conv op must satisfied: stride==1 or (stride==2 and kernel_size==1).
        Set padding=kernel-1, and it just as same as convert_conv_forward_fm.
        
        Params:
            tensor: input feature map
            kernel_size: zeros out side input fmap
        
        Returns:
            frames: list of 'Frame' object
        """
        
        frames = self.convert_conv_forward_fm(tensor,
                                            kernel_size=kernel_size,
                                            padding=kernel_size-1)
        return frames

    def convert_conv_backward_stride_2_kernel_3_fm(self,tensor):
        """ Convert 'output gradient' of a conv layer into frame.

        The conv op must satisfied: stride==2, kernel_size==3.
        Set padding=kernel-1, and it just as same as convert_conv_forward_fm.
        
        Params:
            tensor: input feature map
        
        Returns:
            frames: list of 'Frame' object
        """
        tensor = padding_inside(tensor,padding=1) # inside_padding=stride-1=1
        frames = self.convert_conv_forward_fm(tensor,
                                            kernel_size=kernel_size,
                                            padding=2)#padding=kernel_size-1=2
        return frames
    
    def convert_conv_backward_wm(self,tensor,kernel_size=3,padding=0,stride=1):
        tensor = np.transpose(1,0,2,3)
        return self.convert_conv_forward_wm(tensor)

    def convert_conv_wu_fm(self,tensor):
        pass


def test_convert_conv_feature():
    """ Test 'convert_conv_feature' method
    """
    frame_factory = FrameFactory(tile_len=4)
    feature = np.arange(0.0,50.0).reshape(1,2,5,5).astype(np.half)
    frames = frame_factory.convert_conv_forward_fm(feature,kernel_size=1,padding=0)
    for index,frame in enumerate(frames):
        print(f"------------------ Frame {index} ------------------")
        print(frame.to_str())

def test_convert_conv_forward_stride_2_kernel_3_fm():
    """ Test 'convert_conv_feature' method
    """
    frame_factory = FrameFactory(tile_len=6)
    feature = np.arange(0.0,128.0).reshape(1,2,8,8).astype(np.half)
    frames = frame_factory.convert_conv_forward_stride_2_kernel_3_fm(feature,padding=0)
    for index,frame in enumerate(frames):
        print(f"------------------ Frame {index} ------------------")
        print(frame.to_str())

def test_convert_conv_forward_stride_2_kernel_1_fm():
    frame_factory = FrameFactory(tile_len=2)
    feature = np.arange(0.0,128.0).reshape(1,2,8,8).astype(np.half)
    print(feature)
    frames = frame_factory.convert_conv_forward_stride_2_kernel_1_fm(feature,padding=0)
    for index,frame in enumerate(frames):
        print(f"------------------ Frame {index} ------------------")
        print(frame.to_str())

def test_convert_conv_weight():
    """ Test 'convert_conv_weight' method
    """
    frame_factory = FrameFactory(tile_len=4)
    weight = np.arange(1.0,65.0).reshape(8,2,2,2).astype(np.half)
    print(weight)
    frame_factory.convert_conv_weight(weight)

def test_convert_linear_forward_fm():
    frame_factory = FrameFactory(tile_len=6)
    feature = np.arange(0.0,80.0).reshape(5,16).astype(np.half)
    frames = frame_factory.convert_linear_forward_fm(feature)
    print(frames[0])
    for index,frame in enumerate(frames):
        print(f"------------------ Frame {index} ------------------")
        print(frame.to_str())

def test_convert_linear_backward_fm():
    pass
def test_convert_linear_wg_fm():
    pass
if __name__=="__main__":
    # test_convert_conv_forward_stride_2_kernel_3_fm()
    # test_convert_conv_forward_stride_2_kernel_1_fm()
    test_convert_linear_forward_fm()