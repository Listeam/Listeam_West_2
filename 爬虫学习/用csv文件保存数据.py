import csv
headers = ['name','score','class']
values = [('张三','40','2'),('李四','86','5'),('红红','69','9')] #不同行用元组形式区分开，元组内一个逗号代表一个单元格，write的时候按元组顺序一行一行写

with open(r'C:\Users\Lst12\Desktop\成绩.csv','w',newline='',encoding='utf-8-sig') as f: #在桌面以w形式打开文件，规定新的一行用空白隔开，用utf8解码
    writer = csv.writer(f,delimiter='|')  #创建writer对象，用writer功能把保存的文件写进csv里面，并规定分隔符,默认逗号，如果输入内容有逗号，以区分才需要修改分隔符，如果用excel打开就不用改，因为会自动识别
    writer.writerow(headers) #writerow写一行，把headers写进去
    writer.writerows(values) #writerows写多行，把values写进去

#把字典写进csv
# writer = csv.DictWriter(f,headers,delimiter='|') #第二个参数为表头，字典特有
# writer.writeheader() #字典特有，写入表头函数,基本就差在这一步，本来要writerow现在只要在参数里规定，直接运用函数即可
# writer.writerows(values)