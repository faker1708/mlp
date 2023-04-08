

# x = float(2)



# xx = float('inf')


# print(xx)

# yy = float('nan')
# print(yy)


import torch


ck = torch.cuda.is_available()

print(ck)



a = torch.tensor([[3.,8],[4,2],[7,9],[9,22]])#.cuda()
print(a)
print(a.dtype)

# a = a.half()
# print(a)
a.type(torch.HalfTensor).cuda()

a.dtye = torch.float16
# a.dtye = torch.int8

print(a.dtype)

a = a + a

print(a)

