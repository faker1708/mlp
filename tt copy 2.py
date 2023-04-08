import mlp


import torch


class tt():

    def main(self):
        
        # ll = [4,14,3,2,2]
        # mlp_architecture = [4,2,2,2]
        mlp_architecture = [4,2]
        # lr = 1
        
        nn = mlp.mlp(mlp_architecture)
        # print(a)


        x =torch.tensor([[3.,8],[4,2],[7,9],[9,22]]).cuda().half()
        y_ref = torch.tensor([[ 3,7],    [4,1]], device='cuda:0').half()
        

        nn.train(x,y_ref)
        # nn.update()

        x =torch.tensor([[3.],[4],[7],[9]]).cuda().half()
        y = nn.test(x)
        print(y)


if __name__ == '__main__':
    tt().main()