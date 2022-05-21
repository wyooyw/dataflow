import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import random

class Rectangle:
    def __init__(self,x_range,height,color,tensor=None):
        self.x_range = x_range
        self.height = height
        self.color = color
        self.tensor = tensor
        pass


class RectanglePainter:
    def __init__(self):
        self.color = {
            "red":[1,0,0],
            "green":[0,1,0],
            "blue":[0,0,1]
        }
        self.x_lim = [0,1]
        self.y_lim = [0,1]
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        pass

    def set_lim(self,x_lim,y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim

    def paint(self,range_x,range_y,color="green"):
        """绘制矩形
        range_x为矩形在横轴上的范围
        range_y为矩形在纵轴上的范围
        """
        x1,x2 = range_x
        y1,y2 = range_y
        assert color in self.color,f"Color is not exist:{color}"
        assert x1>=0 and x2>x1 and y1>=0 and y2>y1,f"Error! range_x={range_x},range_y={range_y}"
        plt.gca().add_patch(plt.Rectangle(xy=(x1,y1),
        width=x2-x1, 
        height=y2-y1,
        edgecolor=self.color[color],
        fill=False, linewidth=1))
    def show(self):
        plt.xlim(*self.x_lim)
        plt.ylim(*self.y_lim)
        plt.show()
    def save(self,path="test"):
        plt.xlim(*self.x_lim)
        plt.ylim(*self.y_lim)
        plt.savefig(f"{path}.png",dpi=300) 

class RectangleManager:
    def __init__(self,name="normal"):
        self.rectangle_list = []
        self.rectangle_opt_list = []
        self.rectangle_fix_list = []
        self.x_max = 0
        self.y_max = 0
        self.painter = RectanglePainter()
        self.name = name

    def add_rectangle(self,rectangle):
        self.rectangle_list.append(rectangle)
        # self.rectangle_opt_list.append(rectangle)
        if rectangle.x_range[1] > self.x_max:
            self.x_max = rectangle.x_range[1]

    def add_rectangle_fix(self,rectangle):
        self.rectangle_list.append(rectangle)
        # self.rectangle_fix_list.append(rectangle)
        if rectangle.x_range[1] > self.x_max:
            self.x_max = rectangle.x_range[1]

    def layout(self):
        self.name="resnet-random-feature-grad"
        ground = [0]*(self.x_max+1)

        def put_rec(rec):
            x_range = rec.x_range
            height = rec.height
            ground_max = self.max(ground,x_range[0],x_range[1])
            ground_new_height = ground_max+height
            for i in range(x_range[0],x_range[1]):
                ground[i] = ground_new_height
            rec.y_range = [ground_max,ground_new_height]
            return ground_new_height
            # self.y_max = max(self.y_max,ground_new_height)
        for rec1 in self.rectangle_fix_list:
            new_height = put_rec(rec1)
            self.y_max = max(self.y_max,new_height)
        
        # random.shuffle(self.rectangle_list)
        for rec in self.rectangle_opt_list:
            if rec.tensor.storage.type=="WEIGHT_GRAD":
                new_height = put_rec(rec)
                self.y_max = max(self.y_max,new_height)
        # self.rectangle_opt_list.reverse()

        min_height = 9999999999
        min_order = []
        ground_copy = [*ground]
        for i in range(10):
            y_max_tmp = self.y_max
            random.shuffle(self.rectangle_opt_list)
            for rec in self.rectangle_opt_list:
                if not rec.tensor.storage.type=="WEIGHT_GRAD":
                    new_height = put_rec(rec)
                    y_max_tmp = max(y_max_tmp,new_height)
            print(f"min:{min_height}, got:{y_max_tmp}")
            if y_max_tmp<min_height:
                min_order = [*self.rectangle_opt_list]
                min_height = y_max_tmp
            ground = [*ground_copy]
                
        self.rectangle_opt_list = min_order
        for rec in self.rectangle_opt_list:
                if not rec.tensor.storage.type=="WEIGHT_GRAD":
                    new_height = put_rec(rec)
                    self.y_max = max(self.y_max,new_height)

        return self.y_max
    
    def max(self,ls,range_begin,range_stop):
        m = -999
        for i in range(range_begin,range_stop):
            m = max(m,ls[i])
        return m

    def title(self):
        B = self.y_max
        KB = int(B/1024*100)/100
        MB = int(KB/1024*100)/100
        plt.title(f'{self.name} ({B}B={KB}KB={MB}MB)', fontsize=10)
    
    def paint(self):
        for rec in self.rectangle_list:
            self.painter.paint(rec.x_range,rec.y_range,rec.color)
        self.title()
        # self.y_max += 1
        # self.x_max += 1
        #batch_size=1: 15915800
        #batch_size=4: 39172000
        #batch_size=8: 89701400
        self.painter.set_lim(x_lim=(0,self.x_max+1), y_lim=(0,max(self.y_max+1,89701400)))
        self.painter.show()
    
    def save(self):
        for rec in self.rectangle_list:
            self.painter.paint(rec.x_range,rec.y_range,rec.color)
        self.title()
        # self.y_max += 1
        # self.x_max += 1
        self.painter.set_lim(x_lim=(0,self.x_max+1), y_lim=(0,max(self.y_max+1,89701400)))
        self.painter.save(self.name)

def getRectangleManager():
    # manager = RectangleManagerOrigin()
    # manager = RectangleManagerSimpleReuse()
    # manager = RectangleManagerWeightUpdateImm()
    manager = RectangleManagerRandomSmall()
    return manager

class RectangleManagerOrigin(RectangleManager):
    """最原始的内存布局方法
    各区域分开，不复用任何内存
    """
    def __init__(self):
        super().__init__(name="1.origin")
    
    def layout(self):
        feature_list = []
        weight_grad_list = []
        feature_grad_list = []
        for rec in self.rectangle_list:
            if rec.tensor.storage.type=="ACTIVATION":
                feature_list.append(rec)
            elif rec.tensor.storage.type=="WEIGHT_GRAD":
                weight_grad_list.append(rec)
            elif rec.tensor.storage.type=="FEATURE_GRAD":
                feature_grad_list.append(rec)

        ground = [0]*(self.x_max+1)

        def put_rec(rec):
            x_range = rec.x_range
            height = rec.height
            ground_max = self.max(ground,x_range[0],x_range[1])
            ground_new_height = ground_max+height
            for i in range(x_range[0],x_range[1]):
                ground[i] = ground_new_height
            rec.y_range = [ground_max,ground_new_height]
            return ground_new_height

        #先放feature
        for rec in feature_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)
            for i in range(0,self.x_max+1):
                ground[i] = self.y_max

        #再放weight_grad
        for rec in weight_grad_list:
            rec.x_range = (rec.x_range[0],self.x_max) #设weight_grad的生命周期要一直到最后
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)
            for i in range(0,self.x_max+1):
                ground[i] = self.y_max
        

        #最后放feature_grad
        for rec in feature_grad_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)
            for i in range(0,self.x_max+1):
                ground[i] = self.y_max

        return self.y_max

class RectangleManagerSimpleReuse(RectangleManager):
    """简单地进行内存复用，然各区域间有融合
    """
    def __init__(self):
        super().__init__(name="2.simple-reuse")
    
    def layout(self):
        feature_list = []
        weight_grad_list = []
        feature_grad_list = []
        for rec in self.rectangle_list:
            if rec.tensor.storage.type=="ACTIVATION":
                feature_list.append(rec)
            elif rec.tensor.storage.type=="WEIGHT_GRAD":
                weight_grad_list.append(rec)
            elif rec.tensor.storage.type=="FEATURE_GRAD":
                feature_grad_list.append(rec)

        ground = [0]*(self.x_max+1)

        def put_rec(rec):
            x_range = rec.x_range
            height = rec.height
            ground_max = self.max(ground,x_range[0],x_range[1])
            ground_new_height = ground_max+height
            for i in range(x_range[0],x_range[1]):
                ground[i] = ground_new_height
            rec.y_range = [ground_max,ground_new_height]
            return ground_new_height

        #先放feature
        for rec in feature_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)
        
        #再放weight_grad
        for rec in weight_grad_list:
            rec.x_range = (rec.x_range[0],self.x_max) #设weight_grad的生命周期要一直到最后
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)

        #最后放feature_grad
        for rec in feature_grad_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)

        return self.y_max

class RectangleManagerWeightUpdateImm(RectangleManager):
    """参数立即更新，减小权重的生命周期
    """
    def __init__(self):
        super().__init__(name="3.weight-update-imm")
    
    def layout(self):
        feature_list = []
        weight_grad_list = []
        feature_grad_list = []
        for rec in self.rectangle_list:
            if rec.tensor.storage.type=="ACTIVATION":
                feature_list.append(rec)
            elif rec.tensor.storage.type=="WEIGHT_GRAD":
                weight_grad_list.append(rec)
            elif rec.tensor.storage.type=="FEATURE_GRAD":
                feature_grad_list.append(rec)

        ground = [0]*(self.x_max+1)

        def put_rec(rec):
            x_range = rec.x_range
            height = rec.height
            ground_max = self.max(ground,x_range[0],x_range[1])
            ground_new_height = ground_max+height
            for i in range(x_range[0],x_range[1]):
                ground[i] = ground_new_height
            rec.y_range = [ground_max,ground_new_height]
            return ground_new_height

        #先放feature
        for rec in feature_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)
        
        #再放weight_grad
        for rec in weight_grad_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)

        #最后放feature_grad
        for rec in feature_grad_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)

        return self.y_max

class RectangleManagerRandomSmall(RectangleManager):
    """参数立即更新，减小权重的生命周期
    """
    def __init__(self):
        super().__init__(name="4.random-small")
    
    def layout(self):
        feature_list = []
        weight_grad_list = []
        feature_grad_list = []
        life_short_list = []

        for rec in self.rectangle_list:
            if rec.tensor.storage.type=="WEIGHT_GRAD":
                weight_grad_list.append(rec)
            elif rec.tensor.life_end-rec.tensor.life_begin<=3 or rec.tensor.storage.type=="FEATURE_GRAD":
                life_short_list.append(rec)
            elif rec.tensor.storage.type=="ACTIVATION":
                feature_list.append(rec)
            # elif rec.tensor.storage.type=="FEATURE_GRAD":
            #     feature_grad_list.append(rec)

        ground = [0]*(self.x_max+1)

        def put_rec(rec):
            x_range = rec.x_range
            height = rec.height
            ground_max = self.max(ground,x_range[0],x_range[1])
            ground_new_height = ground_max+height
            for i in range(x_range[0],x_range[1]):
                ground[i] = ground_new_height
            rec.y_range = [ground_max,ground_new_height]
            return ground_new_height

        #先放feature
        for rec in feature_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)
        
        #再放weight_grad
        for rec in weight_grad_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)

        #最后放feature_grad
        min_height = 9999999999
        min_order = []
        ground_copy = [*ground]
        for i in range(10):
            y_max_tmp = self.y_max
            random.shuffle(life_short_list)
            for rec in life_short_list:
                new_height = put_rec(rec)
                y_max_tmp = max(y_max_tmp,new_height)
            print(f"min:{min_height}, got:{y_max_tmp}")
            if y_max_tmp<min_height:
                min_order = [*life_short_list]
                min_height = y_max_tmp
            ground = [*ground_copy]
                
        life_short_list = min_order
        for rec in life_short_list:
            new_height = put_rec(rec)
            self.y_max = max(self.y_max,new_height)

        return self.y_max