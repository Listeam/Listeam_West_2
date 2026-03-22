import pandas as pd

dict_st = { 'a':0.77, 'b':1, 'c':3.5, 'd':5.4, 'e':8.8 } 
new_dict = pd.Series(dict_st) #panda的Series方法把字典转化为更简洁的形式
# print(new_dict)

k = ['a','b','c','d','e']
v = [0.77, 1, None, 44, 61]
new_dict = pd.Series(v,index=k) #也可以用列表或数组或张量形式存储键值以参数的形式给Series
# print(new_dict.values)  #一般都以numpy数组形式输出键值
# print(new_dict.index)   #获取key只能用index不能用keys
# print(new_dict)

"""创建二维对象"""
k = ['1号','2号','3号','4号','5号']
v1 = ['男','女','女','男','女']
v2 = [65,55,23,78,84]
d1 = pd.Series(v1,index=k)
d2 = pd.Series(v2,index=k)

# print(d1,d2,sep='\n')
"""字典创建法"""
d = pd.DataFrame({"性别":v1,"年龄":v2},index=k) #通过DataFrame方法拼接两个键相同的字典，形成简洁表格
# print(d)

"""数组创建法"""
v = [["男",None],['女',55],[None,23],['男',78],["女",84]]
k = ['1号','2号','3号','4号','5号']
c = ["性别","年龄"]
d = pd.DataFrame(v,index=k,columns=c)  #直接将键，列类名，和键对应的所有值以嵌套列表形式，作为参数给DataFrame方法
# print(d)
# print(len(d)) #元素长度为元素个数即index数，跟值无关

"""二维对象的索引"""
# print(d['年龄'])  #绕过index直接显性索引第二个column，也会自动输出元素名即index,但是保存的是整数形式，可以参与运算
# print(d.loc['3号','年龄'])  #二维字典对象访问必须加上loc索引器,loc是显示索引才用，若用隐式索引即用索引值，要用iloc
# print(d.iloc[2,1])  #若直接索引一号的索引值，后面所有values都会出来
# print(d.iloc[[0,2],[0,0]])  #若后索引值相同，会输出两次这个col
# print(d.iloc[[0,2],[1,0]]) #第一个列表代表取第几个元素，第二个列表代表取元素的第几个col，即使没有索引第三个元素的第二个col，也会因为前面索引的那个元素索引了，而同时放出来
# print(d.iloc[2,:])#切片第三个元素的所有col
# print(d.iloc[:,0]) #切片所有元素第一个col

"""对象的变形"""
#.T转置符，用于处理column和index颠倒的情况
# print(d.iloc[:,::-1])   #对 值和对应column取左右颠倒
# print(d.iloc[::-1,:])   #对index取顺序颠倒

"""对象的拼接"""
#concat方法与concatenate语法相似
# print(pd.concat([d1,d2]))   #panda中的字典放弃了常规字典不可重复键名的特性，合并可能出现多个相同index或columns
# print(d.columns.is_unique)  #is_unique函数检查是否有重复值
# print(d.isnull())    #isnull()函数检查是否有缺失值
# print(d.notnull())   #检查非空，不如直接在isnull语句前比如d前加一个非号"~"

# d['属性'] = [0,1,0,1,1]  #添加新元素或新列名
# d.loc['6号'] = ['男',46,1]   #只有索引行的时候需要loc，然后如果是隐性就iloc，索引col不需要
# print(d)

"""去除缺失值"""
v2 = [["是",'厦门'],['否','泉州'],[None,'福鼎'],['是','温州'],["否",'长乐']]
c2 = ['是否爱steam','居住地']
d2 = pd.DataFrame(v2,index=k,columns=c2)
# print(d2)
d3 = pd.concat([d,d2],axis=1)   #二维对象的合并最好用列维度，因为col无论10都会先合并，如果用0就会导致列的堆叠而缺失值
# print(d3)

"""去除缺失值"""
# print(new_dict.dropna())  #dropna函数去除缺失值对应项，默认axis为0，哪一行有空值就整行删除
# print(d.dropna(axis=0,how='all'))    #axis=1即列维度，哪一列有空值就整列删除,how参数后面只能跟any或者all，代表有一个或者全部为缺失值才删除行

# print(d3.dropna(axis=0,thresh=3)) #thresh即要求至少要有几个有效值，输入3，也就是如果出现1个以上缺失值就删除这一行，不包含1
#如果是行维度，一行有四个值，以4为总数，如果是列维度一列有五个，就以5为总数

"""填充缺失值"""
# print(new_dict.fillna(0))  #用0填充缺失值
# print(new_dict.fillna(method='ffill'))  #method参数规定用什么填充，front fill 前值填充缺失值
# print(new_dict.fillna(method='bfill'))  #behind fill 用后值填充缺失值
import numpy as np
# print(new_dict.fillna(np.mean(new_dict)))  #引入np取均值填充

"""导入Excel文件"""
#read方法，注意将excel文件引入pandas必须要加上column行和index列
d4 = pd.read_csv(r'c:\Users\Lst12\Desktop\每日天气数据.csv',index_col=0)  #index_col参数规定csv的第一列为行索引(把column名往右移一格)
# print(d4.head())   #head函数取前五行

"""数据处理分析"""
# print(d4.max())   #算每个column下的最大值，pandas里的聚合函数都是numpy里面的安全版本。自动忽略缺失值
# print(d4.describe())  #describe函数一键输出所有聚合函数 对 所有数学数据 处理后的值，带百分号的是百分位数类比中位数
# print(d4['每日最高温度-2m'].value_counts)统计此column下的value有哪些
"""当需要根据变量对某一事件发生概率影响时一般用.pivot_table()函数,数据透视表，在数据中,发生一般为1,不发生0.则输出其在某因素下的平均值时就是它发生的概率"""
# d.pivot_table(事件column名,index=[],columns=[]) #规定column列和index行有哪些因素，多个因素就用列表储存
#如果因素是具体数值，比较分散，就需要划定范围，比如年龄，如下
#age = pd.cut(d["年龄"],[0,15,30,45,60])  #即按照后参数的范围划分年龄段，以此方式将分散的数据能够写进数据透视表的因素行列

"""merge语法"""

d5 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [101, 102, 101, 103]
})

d6 = pd.DataFrame({
    'dept_id': [101, 102, 104],
    'dept_name': ['HR', 'IT', 'Finance']
})

result = pd.merge(d5, d6, on='dept_id', how='right',suffixes=('_left', '_right'))#merge会自动对应，但需要两个有相同的列名，第一步融合column，第二步确定相同列名
#on参数即用于连接的列名，他会判断，101就是Alice,就是HR，104就是finance，但事实没有人对应104
#how参数，inner为默认，内连，即只找on的共同value，并只取共同部分对应的其他键值
#left左连接，以左on的value为准，左on的103就找不到对应deptname,显示NaN
#right右连接，以右on的value为准，Alice和Charile刚好都对应101，所以Charile也会被拉出来，此处104对应finance，仍然没有人对应
#outer外连，类似直接concat，不取任何相同部分也不偏向一方，相同列全都取，但是还是多一个自动识别 对应相同value的其他列value 的功能，比如concat时，合并后第一个dataframe的deptname全都为空因为第一个本身就没有规定deptname，但merge会自动根据第二个dataframe去对应，101HR...
#left_on和right_on是当value类型相同，但是列名不一样，时的合并方法
#suffixes是当左右两边有相同列名，但是并非连接列，即数据不是同类型，会加上相应后缀
#result = pd.concat([d5,d6])  #普通的concat只是合并。并没有融合为一体，只是生硬的拼接，会有大量缺失值
print(result)

"""groupby语法"""
data = {
    'Department': ['HR', 'IT', 'HR', 'Finance', 'IT', 'Finance', 'HR', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Salary': [50000, 70000, 55000, 60000, 75000, 65000, 52000, 72000],
    'Age': [25, 30, 28, 35, 32, 40, 26, 33],
    'Year': [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024]
}

df = pd.DataFrame(data)
# print(df)
grouped = df.groupby(['Department'])  #分组后的结果是直接打印不出来的，以元组形式储存，有几种department就有几个元组
# for name, group in grouped:  #一个元组第一个元素是department的对应元素，第二个元素就是有关该部门的dataframe，所以能直接一次解包两个变量
#     print(f"{name[0]} 部门:")
#     print(group)

grouped2 = df.groupby(['Department','Year'])  #按department和year分组，即元组的第一个元素有两个元素，一个是部门一个是年份，
# for (name,year), group in grouped2:  
#     print(f"\n{name} {year}-部门:")
#     print(group)

grouped3 = df.groupby(['Department','Year'])['Salary'].agg(['mean', 'median', 'min', 'max', 'std', 'count'])
#经过对salary聚合函数，可以直接打印出来，每个年份的每个部门对应工资的计算结果
print(grouped3)
multi_grouped = df.groupby(['Department','Year']).agg({
    'Salary': ['mean', 'max', 'sum'],
    'Age': ['mean', 'min', 'max'],
    'Employee': 'count'
})
#print(multi_grouped)  #如果要对多个数据分类计算，以字典形式在agg函数参数里列出对应列名和要用的聚合函数