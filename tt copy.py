import mlp


import torch


class tt():

    def main(self):
        ll = [4,2]
        lr = 0.03
        nn = mlp.mlp(ll,lr)

        # print(a)

        x =torch.tensor([[3.],[4],[7],[9]]).cuda()

        y_ref = torch.tensor([[ 3.3],    [4.4]], device='cuda:0')


        ep = 0
        while(1):
            y = nn.forward(x)



            loss = nn.loss_f(y,y_ref)
            loss.backward()
            nn.update()

            fl = float(loss)
            
            print(fl)

            if(fl>2**20):
                raise(BaseException('loss 太大，有问题'))
            # print('yyyyyy',y)

            # w_list = nn.param['w_list']
            # w0 = w_list[0]
            # print('w0_grad',w0.grad)

            # print('\n\n')
            

            ep+=1
            # if(ep>=5):
            #     break
        


if __name__ == '__main__':
    tt().main()