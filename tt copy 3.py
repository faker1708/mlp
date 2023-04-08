import mlp


import torch


class tt():

    def main(self):
        
        # ll = [4,14,3,2,2]
        # mlp_architecture = [4,2,2,2]
        

        in_size = 4
        out_size = 2
        
        mlp_architecture = [4,2]
        nn = mlp.mlp(mlp_architecture)
        # print(a)

        nnb = mlp.mlp(mlp_architecture)
        # nn = nnb

        batch_size = 4

        x = torch.normal(0,10,(in_size,batch_size))

        # x =torch.tensor([[3.,8],[4,2],[7,9],[9,22]])
        # y_ref = torch.tensor([[ 3,7],    [4,1]])
        # y_ref = torch.normal(0,1,(out_size,batch_size))
        y_ref = nnb.test(x)


        nn.train(x,y_ref)
        # nn.update()

        # x =torch.tensor([[3.],[4],[7],[9]])
        y = nn.test(x)
        # print(y)

        

        # x =torch.tensor([[3.,8],[4,2],[7,9],[9,22]])
        y = nn.test(x)
        # print(y)
        print(y-y_ref)
        


if __name__ == '__main__':
    tt().main()