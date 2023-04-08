import mlp


import torch


class tt():

    def main(self):

        in_size = 4
        out_size = 2
        batch_size = 1
        
        mlp_architecture = [in_size,out_size]
        nn = mlp.mlp(mlp_architecture)


        nnb = mlp.mlp(mlp_architecture)


        x = torch.normal(0,10,(in_size,batch_size))

        y_ref = nnb.test(x)


        nn.train(x,y_ref)
        # nn.update()

        # x =torch.tensor([[3.],[4],[7],[9]])
        y = nn.test(x)
        # print(y)

        

        # x =torch.tensor([[3.,8],[4,2],[7,9],[9,22]])

        batch_size = 11
        x = torch.normal(0,10,(in_size,batch_size))

        y = nn.test(x)
        y_ref = nnb.test(x)
        print(y)
        # print(y-y_ref)
        


if __name__ == '__main__':
    tt().main()