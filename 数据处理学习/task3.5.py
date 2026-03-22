import numpy as np
grayscale_image = np.random.randint(0,256,(200,300)) #生成200行300列的随机灰度图像矩阵，元素值在0-255之间,相当于创造这么多个像素点，后面给每个像素点弄三个通道，不同数值的组合即不同颜色

"""将二维灰度图像转成有三个通道RGB的三维彩色图像"""
color_image = np.stack([grayscale_image]*3,axis=-1)  
#stack函数沿指定方向堆叠，第二维度就是把 要堆叠的每个数组的 相同位置的 数组 合并成一个大数组，堆叠几次一个大数组就有几个小数组
#第三维度堆叠可以理解为每个最小维度元素复制n次形成一个n元数组作为新数组的最小维度元素
#把灰度图像矩阵乘3放列表里，相当于有三个相同的矩阵堆叠，且每个最小维度元素堆叠三次，新数组形状为（200，300，3），形成200行300列3通道的彩色矩阵,三通道一开始相等数值

"""应用棕褐色滤镜,不改变形状但改变元素值"""
sepia_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],    #每一行都相当于一个像素点的三个通道的权重，比如像素点(100，150，200)，本来100的位置变成100*0.393 + 150*0.769 + 200*0.189，150的位置变成...以此类推，每个通道都以不同权重构成新通道
    [0.272, 0.534, 0.131]
])
sepia_image = color_image.reshape(60000, 3).dot(sepia_matrix.T) #先把三维数组转为二维数组，需求是每个像素点的三个通道分别和滤镜矩阵的每一行三个系数相乘再相加，并成新的一行，得到新的三个通道值，再把二维数组转回三维数组
sepia_image = sepia_image.reshape(200,300,3)  #滤镜后的三维数组

"""切片限制数组元素范围在0-255之间,并转为无符号整数类型"""
limited_sepia_image = np.clip(sepia_image,0,255).astype(np.uint8) #clip函数限制数组元素范围在0-255之间，大于的变成255，矩阵乘法产生大量浮点，astype函数把浮点数组转为无符号整数类型（uiny8），范围0-255

"""增强图像饱和度,用过饱和公式"""
saturation_factor = 1.5 #饱和度因子，0为黑白，1为原图，>1为更鲜艳
lumi_factor = np.array([0.299,0.587,0.114])  #亮度系数

L_matrix = limited_sepia_image.reshape(60000,3).dot(lumi_factor.T)  #先把三维数组转为二维数组，每个像素点的三个通道分别和对应的权重相乘再相加，得到每一个通道的亮度值，化3为1得新矩阵后面再复制3次,因为同一个通道减的亮度一样
L_matrix = L_matrix.reshape(200,300,1)  #再把二维数组转回三维数组，最后一个维度只有一个元素，即亮度值
L_matrix = np.repeat(L_matrix,3,axis=-1) #用repeat不增加维度的同时把亮度值矩阵的最小维度元素复制三次，形成三个通道的亮度矩阵,方便后面公式减法运算

#饱和度增强公式 C_new = L + alpha * (C_old - L)
extra_saturation_matrix = L_matrix + saturation_factor * (limited_sepia_image - L_matrix)  

"""切片限制数组元素范围在0-255之间,并转为无符号整数类型"""
limited_extra_saturation_image = np.clip(extra_saturation_matrix,0,255).astype(np.uint8)  #限制范围并转为无符号整数类型，得到色彩增强的像素点矩阵


"""添加渐变边框"""
border_size = 20
border_image = np.zeros((200,300+border_size*2,3))  #生成全0矩阵，再直接把原矩阵搬进去，方便后续通过循环给每行的边框像素定位
border_image[:,20:320,:] = color_image  #原图像放进去

gradient_factor = np.linspace(0,1,border_size).reshape(1,border_size,1)  #渐变因子，作为后面起终颜色的权重

left_edge = color_image[:,0]   #取每一行最左边和最右边的边界像素点
right_edge = color_image[:,-1]

left_border = (1-gradient_factor)*0 + left_edge[:,np.newaxis,:]*gradient_factor   #利用广播，渐变因子已经转为20行的列向量，左乘它，会让左边界像素点复制20次形成(200,20,3)的矩阵，再和(200,20,3)的黑色矩阵相加，得到左边框渐变矩阵
right_border = right_edge[:,np.newaxis,:]*(1-gradient_factor) + gradient_factor*255   #利用向量运算，提高效率，直接得到一整个边框矩阵，后面直接像前面一样填进全0矩阵

border_image[:,:20,:] = left_border
border_image[:,320:,:] = right_border

border_image = np.clip(border_image,0,255).astype(np.uint8)


"""for循环做法,一开始只能理解这个....."""
# for row in range(200):   #以大括号为标准，不太好给每个像素点都弄边框，就给每一行即300个像素点弄边框，所以需要两百行
#     left_border = color_image[row,0]   #最左边和最右边的边界像素点，作为左渐变的终止颜色，和右渐变的起始颜色
#     right_border = color_image[row,299]
#     #左边框渐变
#     for col,weight in enumerate(gradient_factor):  #同时求索引和权重，索引用于定位边框位置，权重用于决定该边框像素的渐变程度
#         gradient_color = (1-weight)*0 + weight*left_border    #渐变公式，起始颜色为0即黑色，意为从黑色渐变到原始色，原始色的占比越来越大直到变成原始色
#         border_image[row,col,:] = gradient_color  #col到19,打印宽度为20像素的边框
#     #右边框渐变
#     for col,weight in enumerate(gradient_factor):
#         gradient_color = (1-weight)*right_border + weight*255   #渐变公式，起始颜色为原始色，终止颜色为255即白，意为从原始色渐变到白色，原始色的占比越来越小直到变成白色
#         border_image[row,300+border_size+col] = gradient_color  #col从320到339,打印宽度为20像素的边框

print(border_image)  #最终打印出添加渐变边框的彩色图像矩阵
