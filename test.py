from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.target_gen.memory.segment import Segment,SegmentType
from compiler.generate.net.net import Net
from compiler.generate.op.conv import ConvDualGenerator,ForwardConv2d
from compiler.generate.op.relu import ReluDualGenerator,ForwardRelu
from compiler.generate.op.split import SplitDualGenerator
from compiler.generate.op.aggregate import AggregateDualGenerator
from compiler.utils.unique_class_name import unique_class_name
from compiler.graph.merger import Finder
def testMemoryManager():
    mem = MemoryManager()
    mem.calcBases()
    for name,segment in mem.segments.items():
        print(f"name:{name}")
        print(f"\tsegment.base:{segment.base},segment.size:{segment.size}\n")

def testDualGenerator():
    conv_forward,conv_backward = ConvDualGenerator(in_batch=1,
                                                    in_width=4,
                                                    in_height=4,
                                                    in_channels=3,
                                                    out_channels=2,
                                                    kernel_size=2,
                                                    padding=0).getDual()
    relu_forward,relu_backward = ReluDualGenerator(in_batch=1,
                                                    in_width=3,
                                                    in_height=3,
                                                    in_channels=2).getDual()
    relu_forward.add_predecessor(conv_forward)
    relu_backward.add_successor(conv_backward)
    
    # split_forward,split_backward = SplitDualGenerator(in_batch=1,
    #                                                 in_width=3,
    #                                                 in_height=3,
    #                                                 in_channels=2).getDual()

    # split_forward.add_predecessor(relu_forward)
    # split_backward.add_successor(relu_backward)

    conv2_forward,conv2_backward = ConvDualGenerator(in_batch=1,
                                                    in_width=3,
                                                    in_height=3,
                                                    in_channels=2,
                                                    out_channels=2,
                                                    kernel_size=3,
                                                    padding=1).getDual()
    
    conv2_forward.add_predecessor(relu_forward)
    conv2_backward.add_successor(relu_backward)

    net = Net()
    net.ops = [conv_forward,relu_forward,conv2_forward]

    pattern = [ForwardConv2d,ForwardRelu]
    finder = Finder(net,pattern)

    find = finder.find()
    for f in find:
        for item in f:
            print(item)
        print("-----")
    # for op in net.ops:
        # print(op)
    
    # agg_forward,agg_backward = AggregateDualGenerator(in_batch=1,
    #                                                 in_width=3,
    #                                                 in_height=3,
    #                                                 in_channels=2).getDual()
                                                
    # agg_forward.add_predecessor(conv2_forward)
    # agg_forward.add_predecessor(split_forward)

    # agg_backward.add_successor(conv2_backward)
    # agg_backward.add_successor(split_backward)
    
    # conv_forward.print()
    # relu_forward.print()
    # split_forward.print()
    # conv2_forward.print()
    # agg_forward.print()
    # print("------------------------------------------")
    # conv_backward.print()
    # relu_backward.print()
    # split_backward.print()
    # conv2_backward.print()
    # agg_backward.print()

    



def testPointer():
    from compiler.utils.pointer import Pointer
    ptr = Pointer(None)
    ptr2 = ptr
    print(ptr,ptr2)
    ptr.set("helloworld")
    print(ptr,ptr2)

class MyClass:
    def __init__(self):
        # print(unique_class_name(self))
        pass

if __name__=="__main__":
    # testPointer()
    testDualGenerator()
    # testMemoryManager()
    # c1 = MyClass()
    # print(type(c1).__name__)
    # c2 = MyClass()