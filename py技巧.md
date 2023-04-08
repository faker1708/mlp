
inf 与 nan 的获取

inf =float('inf')
nan = float('nan')


# if(fl==inf):
#     break

# 但 fl==nan 是不能判断的。所以，建议不要用== 来判断

# inf 与 nan的判断

if(math.isnan(fl)):
    break
elif(math.isinf(fl)):
    break



向量

x = torch.unsqueeze(x, dim=1)


# 矩阵转置
b_s = b_s.permute(1,0)     





https://blog.csdn.net/weixin_35757704/article/details/115909939


修改类型

a = a.type(torch.float16)


拷贝类型

    in_dtype = x.dtype
    in_device = x.device

    
    y = y.to(in_device)
    y = y.type(in_dtype)


    