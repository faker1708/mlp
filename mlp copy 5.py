


import torch
import math


# 感觉batchsize可以删除了呀，因为需要它的地方也有 lr 可以协同控制。

# 如果batch_size 为1 一定要保证你的输入输出 都是列向量
# x2 = torch.unsqueeze(x1, dim=1)

class mlp():

    def __init__(self,mlp_architecture,mode = 'default'):
        # mlp_architecture = [2,2] # 超参是个列表，记录有几层神经元，每层有几个神经元，列表里的值 不是个数 ，而是个数的对数。具体构造规则见build函数。
        self.mlp_architecture = mlp_architecture
        self.mode = mode
        
        
        


        # self.depth = len(mlp_architecture)
        # self.lr = lr
        self.non_linear = torch.nn.ReLU(inplace=False)   # 定义relu

        self.__build_nn()
    
    def __build_nn(self):
        mlp_architecture=self.mlp_architecture


        depth = len(mlp_architecture)
        w_list = list()
        b_list = list()
        for i,ele in enumerate(mlp_architecture):
            if(i<=depth -2):
                # kn = mlp_architecture[i]
                # km = mlp_architecture[i+1]

                # n = 2**kn
                # m = 2**km
                
                n = mlp_architecture[i]
                m = mlp_architecture[i+1]

                # print('mlp m',m)
                

                if(self.mode == 'cpu'):
                    w = torch.normal(0,1,(m,n)).cpu()
                    b = torch.normal(0,1,(m,1)).cpu()
                elif(self.mode == 'gpu' or self.mode == 'cuda'):
                    w = torch.normal(0,1,(m,n)).cuda()
                    b = torch.normal(0,1,(m,1)).cuda()
                elif(self.mode == 'cuda_half'):
                    w = torch.normal(0,1,(m,n)).cuda()
                    b = torch.normal(0,1,(m,1)).cuda()

                
                # 默认记录计算图
                w.requires_grad=True
                b.requires_grad=True


                w_list.append(w)
                b_list.append(b)
                    

        param = dict()
        param['w_list'] = w_list
        param['b_list'] = b_list
        # param['depth'] = depth


        self.param = param
        # return param

    def __forward(self,x):
        # y = 0

        # if(gr==0):

        mlp_architecture = self.mlp_architecture 
        param = self.param




        w_list= param['w_list']
        b_list= param['b_list']

        # print('__forward')

        depth = len(mlp_architecture)

        for i in range(depth-2):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('__forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.non_linear(w @ x + b)

        w = w_list[depth-2]
        b = b_list[depth-2]
        x = w @ x + b
        # 最后一层不要加非线性，加了relu会导致大量数据梯度为0，网络无法收敛了。


        y = x
        return y

    def __loss_f(self,y,y_ref):

        dd = (y-y_ref)**2 /2
        
        loss = dd.sum()

        return loss
    
    def __update(self):
        # lr batch_size 只是在这里用到，用来梯度下降。
        # 但 batch_size 和lr同时出现 ，所以感觉没必要有batch_size 这个量了。

        mlp_architecture = self.mlp_architecture 
        
        param = self.param
        # batch_size = self.batch_size
        lr = self.lr

        w_list= param['w_list']
        b_list= param['b_list']

        # print('__update')

        with torch.no_grad():
            
            depth = len(mlp_architecture)
            for i in range(depth-1): 
                # print('__update',i)
                w = w_list[i]
                b = b_list[i]

                # w -= lr * w.grad / batch_size
                # print('upd w_grad',w.grad)
                w -= lr * w.grad 
                w.grad.zero_()


                b -= lr * b.grad 
                # b -= lr * b.grad / batch_size
                b.grad.zero_()

    def test(self,x):

        # 测试与训练不同，这是确实要一个值出来 ，并且不要算梯度。


        with torch.no_grad():
            y = self.__forward(x)
        return y

    def train(self,x,y_ref):
        nn = self


        self.lr = 2**-9

        total_cost = 0    # 无论如何，会在这么多轮训练后结束，如果不是正数，则此限制无效。

        precious = 10
        patient = 2**8  # 连续这么多轮训练都不收敛。
        
        ceil_line = 2**20   # 损失的上界


        # 局部变量 草稿 temp
        bad_count = 0  # 坏情况计数 当出现相同时，加一
        old_loss = 0


        ep = 0
        while(1):
            # print('ep',ep)

            y = nn.__forward(x)


            loss = nn.__loss_f(y,y_ref)
            loss.backward()

            nn.__update()

            fl = float(loss)

            if(fl>=old_loss):
                bad_count+=1
            else:
                bad_count = 0  #清零
                # 这里不去监测loss是否增大。
            old_loss = fl
            
            fl= fl**0.5
            print('loss',ep,fl)



            if(fl > ceil_line):
                # raise(BaseException())
                print('loss 太大，有问题.考虑减小学习率')
                break
            elif(fl< 2**-precious ):
                print('精度达标')
                break
            if(math.isnan(fl)):
                print('error,nan')
                break
            elif(math.isinf(fl)):
                print('error,inf')
                break

            ep+=1
            if(total_cost>0):
                if(ep>total_cost):
                    print('超时')
                    break

            if(bad_count>patient):
                print('不收敛了')
                break
            # if(ep>2**13):
            #     break
