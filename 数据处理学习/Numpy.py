import numpy as np

"""同化定理"""
list1 = np.array([1,2,3]) #np的array功能会将传统列表去逗号，并且有严格的分类，整数数组不允许插入浮点苏，浮点数数组不允许插入整数
list1[0] = 100.9  #浮点数被插入整数数组会被自动截断成100
list2 = np.array([1.0,2,3])  #只要有一个浮点数，默认其他都是浮点数
list2[1] = 6  #整数被插入浮点数组会自动加上浮点
# print(list1)
# print(list2)

"""共同改变定理"""
list1 = np.array([5,2,0,1,3,1,4])
list2 = list1.astype(float)  #区别于array，astype属于类里面的功能，而array是个函数，需要加np前缀，astype转化数组类型
# print(list2)
# print(list1+0.0)  #整数数组和浮点数的运算或者除以一个整数，都会转化为浮点型数组
# print(list1+1.0)
# print(list1*1.0)
# print(list1/1)

int_list = np.array([1,2,3])
float_list = np.array([1.0,3,5])
# print(int_list + float_list)   #整数型数组和浮点型数组相加也会转化为浮点型数组

"""数组维度"""
#括号里几个数字就是几维数组，一维数组简称为向量，二维数组简称为矩阵
list1 = np.ones((4))  #一维数组默认为第一个参数，即一个列表里有几个元素
list2 = np.ones((1,2,3))  #ones即生成全是1的浮点数组的方法，默认浮点是为了后续添加有浮点的元素不会被截断
#第一个参数为，最大维度里有几个相同列表元素，第二个参数为第二大维度里有几个相同列表元素，第三个参数为最小维度里有几个相同列表元素
list3 = np.ones((3,2,1))  #最大中括号里有三个列表，三个列表各自有两个小列表，两个小列表里各自有一个元素
# print(list1)
# print(list2)
# print(list3)
# print(list1.shape)  #shape方法获得数组维度
# print(list2.shape)
# print(list3.shape)  

list1 = np.arange(10)  #快捷生成0-9的列表
# print(list1.reshape(2,5))  #reshape方法从一维转为二维，三个参数乘积必然相同，为元素总数，因此如果5位置改为-1，让他按照此逻辑推算，仍会输出一样的结果
# print(list1.reshape(2,-1))
# print(list1.reshape(2,5).reshape(-1)) #再从二维转为一维，直接用-1，自动计算最后一个参数更省力

"""创建指定数组"""
list1 = np.array([[1,2,3]])  #创建行矩阵，只有一个包含多元素的元素的二维数组
list2 = np.array([[1],[2],[3]])  #创建列矩阵，有多个只有一个元素的二维数组
list3 = np.array([[1,2,3],[4,5,6]])  #创建矩阵，有多个包含多个元素的元素的二维数组
# print(list1,list2,list3, sep = "\n")

"""创建递增数组"""
list1 = np.arange(10,20,3.0) #跟range同理,想要转为浮点数组把间隔加上浮点即可
"""创建同值数组"""
list1 = np.ones((4))
list1 = np.zeros((4))  #全0数组
list1 = 3.14*np.ones((4)) #要指定数字直接乘在前头即可

"""创建随机数组"""
list1 = np.random.random((2,5)) #生成0-1均匀分布的浮点型随机数组，参数为数组形状,二行五列
list2 = 40*np.random.random((3,3))+60  #前面参数修改参数范围为0-40，再在后面加60即60-100的数字
list3 = np.random.randint(10,100,(3,4))  #创建整数型随机数组
list4 = np.random.normal(0,1,(2,3))   #创建符合正态分布的随机数组,第一个参数为均值，第二个参数为标准差，第三个为形状
list5 = np.random.randn(2,3)   #如果是0-1正态分布可以直接用randn,只有形状参数
# print(list1)
# print(list2)
# print(list3)
# print(list4)
# print(list5)

"""花式访问数组元素"""
list1 = np.arange(1,10).reshape(3,3)
# print(list1)
# print(list1[[0,1,2],[0,1,2]]) #双层求索，第一个列表元素代表要取的行数，第二个列表代表要取得列数，即取坐标(0,0)(1,1)(2,2)...输出向量
# list1[[0,1,2],[0,1,2]] = 100  #修改每个坐标对应的数为100
# print(list1)

list2 = np.arange(1,21).reshape(4,5)
# print(list2)
# print(list2[1:3,1:-1]) #矩阵切片，截取2，3两行，第二个到倒数第二个
# print(list2[2,:]) #切第三行所有元素也可以不加:简略化，但要注意提取列的时候不能简化
# print(list2[:,2]) #切第三列，每行只提取一个数，则生成向量，节约空间，一定要生成矩阵就用reshape转成二维，若想转为列矩阵也可用.T方法转置
# print(list2[:,1:3]) #切二到三列，生成矩阵

"""数组切片仅是视图""" """数组赋值仅是绑定"""
#切片不会创建一个新的变量，只是把视角移到此部分，如果改变切片元素，原数组元素也会被改变，节省内存，同理把数组赋值给另一个数组，也会相互影响
#要解除这种绑定关系，生成独立数组，对原数组用copy方法创建新变量即可

"""数组的翻转"""
list1 = np.arange(10)
flip_list1 = np.flipud(list1)  #ud即updown,向量翻转即左右顺序翻转
# print(flip_list1)

list2 = np.arange(1,21).reshape(4,5)
fliplr_list2 = np.fliplr(list2)   #lf即leftright,每一行都左右翻转
# print(fliplr_list2)

flipud_list2 = np.flipud(list2)   #ud即updown,列序数全倒
# print(flipud_list2)

"""数组的拼接"""
list1 = np.arange(10)
list2 = np.arange(10,20)
list12 = np.concatenate([list1,list2])  #规定两个数组拼接的形状，此处即合成一个一维数组
# print(list12)   #concatenate方法拼接数组，直接数组相加会变成各个元素相加生成一个新数组
# print(list1.reshape(5,2),list2.reshape(5,2),sep="\n")
list12 = np.concatenate([list1.reshape(5,2),list2.reshape(5,2)],axis = 1)  #拼接数组必须同维度，axis为默认参数为0，0即按行维度来拼接，1按列维度来拼接
# print(list12)  #行维度即压缩每一行个数，即纵向堆叠，按列的方向全部取下来拼接
                 #列维度即压缩每一列个数，横向堆叠，把每个数组的相同行堆叠，再拼接

"""数组的分裂"""
list1 = np.arange(1,11)  #切向量
a,b,c,d,e = np.split(list1,[2,4,6,8])  #一维数组第二个参数就不是行列的索引值了，单纯第几个数字，一刀两断，在索引值处切一刀，2就是在第三个元素的位置前切一刀
# print(a,b,c,d,e,sep="\n")

list1 = np.arange(1,9).reshape(2,4)
a,b = np.split(list1,[1],axis=0)  #二维数组，axis=1列维度，即在第二列前面切一刀
# print(list1)     #axis=0，行维度，第二个参数就是行的索引，即在第二行前面切
# print(a,"\n",b)

"""不同形状数组的运算"""
list1 = np.array([100,0,-100])  #向量的广播，即沿竖直水平方向以重复延申，直到形状跟矩阵一样
list2 = np.random.random((10,3)) #即使是不同形状也要求两数组的最小维度元素数相同
# print(list1*list2)

list1 = np.arange(3).reshape(3,1)  #列矩阵的广播
list2 = np.ones((3,5)) #要求最大维度元素数相同即行数
# print(list1*list2)

list1 = np.array([100,0,-100])
list2 = np.arange(3).reshape(3,1) #向量与列矩阵的运算，相互适应广播，向量根据列矩阵列数向下延伸，列矩阵根据向量行数向右延申
# print(list1*list2)

"""点积法求矩阵乘积"""
list1 = np.arange(15).reshape(3,5)
list2 = np.arange(5)  #要求前矩阵列数等于后矩阵的行数，点积法极其注重顺序
# print(np.dot(list1,list2))  #(3，5)(5，1)得出结果为(3,1)形状的向量，而第i行第j列的元素为前矩阵的i行和后向量一整行乘积和
                            #向量和任何形状乘积结果必然是向量，即使得出的是列矩阵也转化
list3 = np.arange(15).reshape(5,3) #矩阵乘矩阵，第i行第j列的元素为前矩阵的第i行和后矩阵的第j行乘积和
# print(np.dot(list1,list3)) 

"""Numpy的数学函数"""
#np.abs绝对值，np.sin/cos/tan三角函数(注意用pi时要用np.pi),底数**数组即指数函数(注意用e时要用np.exp(数组))
#np.log(数组)默认底数为e，即ln，np.log(数组)/np.log(底数)，即用换底公式换底
# print(list3)
# print(np.max(list3,axis=0))  #有axis参数，不同维度求最值类似数组拼接和分裂,但是函数里axis没有默认行维度，默认就是求整体最值
#mp.sum/prod/mean/std,类似最值，求和，求积，求平均，求标准差
#注意max开始后面都是聚合函数，聚合函数有更安全的版本，即前缀nan，比如nanmax，意为Not a Number,这样如果数据缺失也不影响计算

"""布尔型数组"""
list1 = np.arange(10).reshape(5,2)
# print((list1 >= 5)|(list1 <=2)) #实质是在打印这一步判断是否符合条件，符合就输出true，反之false

list1 = np.random.randn(10000)
num = np.sum(np.abs(list1)<1)  #用abs函数判断绝对值小于一，再用np的sum函数求符合条件的元素总数即求布尔数组的True数量
# print(num)

list1 = np.arange(1,10)
list2 = np.flipud(list1)
# print(np.any(list1 == list2))  #any函数判断，只要两数组存在一个元素相等就返回一个True

list1 = np.random.randn(100000)
# print(np.all(list1 > 250))  #all函数判断，只有list1的全部元素都大于500，才返回True

"""布尔型数组作为掩码"""
list1 = np.arange(1,13).reshape(3,4)
# print(list1[list1 > 4])  #布尔型数组可以作为索引值，即掩码，切片原数组符合条件的元素并生成一个向量

"""获得满足条件的元素位置索引值"""
list1 = np.random.randn(100000)
# print(np.where(list1 == np.nanmax(list1))) #where函数求索最大值元素，输出索引值

"""从数组到张量"""
#PyTorch吸收NumPy语法，语法几乎一致，np对应torch，数组array对应张量tensor，n维数组对应n阶张量

"""三维数组的聚合"""
list_3d = np.array([
    [[1, 2, 3],
     [4, 5, 6]],  

    [[7, 8, 9], 
    [10, 11, 12]],   

    [[13, 14, 15], 
    [16, 17, 18]]    #形状为(3,2,3)
])
# print(np.sum(list_3d,axis=0))  #axis=0即对最外层维度求和，得到一个数组形状为(2,3)，每个元素为最外层括号里 每个括号相同即对齐位置的数字相加。即1+7+13，2+8+14，3+9+15...
# print(np.sum(list_3d,axis=1))  #axis=1即对第二维度求和，得到一个数组形状为(3,3)，每个元素为第二层括号里 每个括号 相同位置的数字 相加，即1+4，2+5，3+6...
# print(np.sum(list_3d,axis=2))  #axis=2即对第三即最内层维度求和，得到一个数组形状为(3,2)，每个元素为最小括号内所有列元素之和，123，456分别变成6，15