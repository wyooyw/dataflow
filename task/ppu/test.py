import numpy as np
import torch
test_index = 1
def load(index,name,shape,type=torch.float32):
    tensor = np.loadtxt(f"test_data/backward/{index}/{name}.txt", delimiter=" ").reshape(shape)
    tensor = torch.from_numpy(tensor)
    return tensor
    
def different(result,answer):
    if result.shape==answer.shape:
        print(torch.max(torch.abs(result-answer)))
        print(torch.max(torch.abs(result-answer))<0.01)
    else:
        print(f"Shape is not equal! result.shape={result.shape}, answer.shape={answer.shape}")
def test1():
    bn_std_reci = load(1,"BackwardBatchnorm.std_reci",1)#.half()
    bn_input_grad = load(1,"BackwardBatchnorm.input_grad",[1,1,4,4])#.half()
    relu_mask = load(1,"BackwardRelu.mask",[1,1,4,4])#.half()
    relu_output_grad = load(1,"BackwardRelu.output_grad",[1,1,4,4])#.half()
    # print(bn_std_reci.dtype)
    rst = torch.mul(relu_mask,relu_output_grad)
    rst = torch.transpose(rst,1,3)
    rst = torch.mul(rst,bn_std_reci)
    rst = torch.transpose(rst,1,3)
    different(rst,bn_input_grad)

def test2():
    bn_std_reci = load(2,"BackwardBatchnorm.std_reci",1)#.half()
    bn_input_grad = load(2,"BackwardBatchnorm.input_grad",[1,1,4,4])#.half()
    relu_mask = load(2,"BackwardRelu.mask",[1,1,4,4])#.half()
    res_output_grad = load(2,"BackwardResAccSingle.output_grad",[1,1,4,4])#.half()
    res_output_grad_res = load(2,"BackwardResAccSingle.output_grad_res",[1,1,4,4])#.half()
    res_std = load(2,"BackwardResAccSingle.std",1)#.half()

    rst = res_output_grad_res
    
    rst = torch.transpose(rst,1,3)
    rst = torch.mul(rst,res_std)
    rst = torch.transpose(rst,1,3)

    rst += res_output_grad

    rst = torch.mul(rst,relu_mask)
    rst = torch.transpose(rst,1,3)
    rst = torch.mul(rst,bn_std_reci)
    rst = torch.transpose(rst,1,3)
    different(rst,bn_input_grad)

def test_double_bn():
    bn_std_reci1 = load(2,"BackwardBatchnorm1.std_reci",1).half()
    bn_input_grad1 = load(2,"BackwardBatchnorm1.input_grad",[4,1,4,4]).half()
    bn_std_reci2 = load(2,"BackwardBatchnorm2.std_reci",1).half()
    bn_input_grad2 = load(2,"BackwardBatchnorm2.input_grad",[4,1,4,4]).half()

    relu_mask = load(2,"BackwardRelu.mask",[4,1,4,4]).half()

    res_output_grad = load(2,"BackwardResAccSingle.output_grad",[4,1,4,4]).half()
    res_output_grad_res = load(2,"BackwardResAccSingle.output_grad_res",[4,1,4,4]).half()
    res_std = load(2,"BackwardResAccSingle.std",1).half()

    rst = res_output_grad_res
    
    rst = torch.transpose(rst,1,3)
    rst = torch.mul(rst,res_std)
    rst = torch.transpose(rst,1,3)

    rst += res_output_grad

    rst = torch.mul(rst,relu_mask)

    rst = torch.transpose(rst,1,3)
    rst1 = torch.mul(rst,bn_std_reci1)
    rst2 = torch.mul(rst,bn_std_reci2)
    rst1 = torch.transpose(rst1,1,3)
    rst2 = torch.transpose(rst2,1,3)
    different(rst1,bn_input_grad1)
    different(rst2,bn_input_grad2)

def test_maxpool():
    input = np.loadtxt(f"test_data/forward/0/detail/ForwardRelu.output.txt", delimiter=" ").reshape(4,1,32,32)
    
    batch,channel,height,width = input.shape
    input = input.reshape(batch,channel,height//2,2,width//2,2)
    input = np.transpose(input,(0,1,2,4,3,5))
    input = input.reshape(batch,channel,height//2,width//2,4)
    output = np.max(input,axis=4).reshape(-1)
    ptr = np.argmax(input,axis=4).reshape(-1)
    
    maxpool_output = np.loadtxt(f"test_data/forward/0/ForwardMaxpool.output.txt", delimiter=" ").reshape(-1)
    maxpool_ptr = np.loadtxt(f"test_data/forward/0/ForwardMaxpool.ptr.txt", delimiter=" ").reshape(-1)
    
    delta_output = output - maxpool_output
    delta_ptr = ptr - maxpool_ptr
    err_output = (np.max(delta_output),np.min(delta_output),np.sum(delta_output>0.001))
    err_ptr = (np.max(argmax - maxpool_ptr),np.min(argmax - maxpool_ptr))
    print("err_output:",err_output)
    print("err_ptr:",err_ptr)

def test_maxpool2():
    input = np.loadtxt(f"test_data/forward/0/detail/ForwardRelu.output.txt", delimiter=" ").reshape(4,32,2,8,2)
    input = np.transpose(input,(0,1,3,2,4))
    input = input.reshape(4,32,8,4)
    max = np.max(input,axis=3).reshape(-1)
    argmax = np.argmax(input,axis=3).reshape(-1)
    
    maxpool_output = np.loadtxt(f"test_data/forward/0/ForwardMaxpool.output.txt", delimiter=" ").reshape(-1)
    maxpool_ptr = np.loadtxt(f"test_data/forward/0/ForwardMaxpool.ptr.txt", delimiter=" ").reshape(-1)
    
    delta_output = max - maxpool_output
    delta_ptr = argmax - maxpool_ptr
    err_output = (np.max(delta_output),np.min(delta_output),np.sum(delta_output>0.001))
    err_ptr = (np.max(argmax - maxpool_ptr),np.min(argmax - maxpool_ptr))
    print("err_output:",err_output)
    print("err_ptr:",err_ptr)

def test_maxpool3():
    input = np.loadtxt(f"test_data/forward/1/detail/ForwardRelu.output.txt", delimiter=" ").reshape(4,1,16,16)
    
    batch,channel,height,width = input.shape
    input = input.reshape(batch,channel,height//2,2,width//2,2)
    input = np.transpose(input,(0,1,2,4,3,5))
    input = input.reshape(batch,channel,height//2,width//2,4)
    output = np.max(input,axis=4).reshape(-1)
    ptr = np.argmax(input,axis=4).reshape(-1)
    
    maxpool_output = np.loadtxt(f"test_data/forward/1/ForwardMaxpool.output.txt", delimiter=" ").reshape(-1)
    maxpool_ptr = np.loadtxt(f"test_data/forward/1/ForwardMaxpool.ptr.txt", delimiter=" ").reshape(-1)
    
    delta_output = output - maxpool_output
    delta_ptr = ptr - maxpool_ptr
    err_output = (np.max(delta_output),np.min(delta_output),np.sum(delta_output>0.001))
    err_ptr = (np.max(delta_ptr),np.min(delta_ptr))
    print("err_output:",err_output)
    print("err_ptr:",err_ptr)

def test4():
    input = np.loadtxt(f"test_data/forward/1/ForwardBatchnorm.input.txt", delimiter=" ").reshape(-1)
    mean = np.loadtxt(f"test_data/forward/1/ForwardBatchnorm.mean.txt", delimiter=" ").reshape(-1)
    std_reci = np.loadtxt(f"test_data/forward/1/ForwardBatchnorm.std_reci.txt", delimiter=" ").reshape(-1)
    output = np.loadtxt(f"test_data/forward/1/ForwardRelu.output.txt", delimiter=" ").reshape(-1)

    predict = (input - mean) * std_reci
    predict = predict * (predict > 0)
    err = output - predict
    print("err:",np.max(err),np.min(err))
    print(np.sum(err>0.1),"/",len(err))
if __name__=="__main__":
    # test_double_bn()
    # test_maxpool()
    # test_maxpool2()
    # test_maxpool3()
    test4()