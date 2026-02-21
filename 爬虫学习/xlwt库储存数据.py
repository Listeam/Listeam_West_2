import xlwt
import random
from lxml import etree   #导入etree模块
import requests
from pprint import pprint
import time
headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
}
workbook = xlwt.Workbook(encoding='utf8') #用xlwt的Workbook方法创建workbook对象，规定用utf8解析
sheet1 = workbook.add_sheet('爬虫方法')  #add_sheet()方法创建工作表并命名
sheet2 = workbook.add_sheet('爬虫成绩')
sheet2.write(0,0,'座号')
for i in range(1,51):
    sheet2.write(i,0,i)
sheet2.write(0,1,'语文')
sheet2.write(0,2,'数学')
sheet2.write(0,3,'英语')
for col in range(1,51):
    for row in range(1,4):
        sheet2.write(col,row,random.randint(60,100))

#以xpath中做的字典为例，展示如何在excel中存储字典
Urls = []
films_info = []
for i in range(0,100,20):
    response = requests.get(f"https://movie.douban.com/review/best/?start={i}",headers = headers)
    text = response.text
    text_html = etree.HTML(text)
    links = text_html.xpath('//h2/a/@href')
    time.sleep(2)   #每次请求间隔2秒，防止被封IP
    for link in links:   #提取每页每个影评的链接
        Urls.append(link)
        
#print(Urls)
i = 0
for url in Urls:
    try:
        response = requests.get(url,headers = headers)
        text = response.text
        text_html = etree.HTML(text)

        name = text_html.xpath('//div[@class="subject-title"]/a/text()')[0][2:]  #提取电影名，发现前面有两个空格字符，所以用切片去掉
        director = text_html.xpath('//*[@id="content"]/div/div[2]/div[4]/div[4]/ul/li[1]/span[2]/text()')[0].strip()
        main_actors = text_html.xpath('//*[@id="content"]/div/div[2]/div[4]/div[4]/ul/li[2]/span[2]/text()')[0].strip()
        type0 = text_html.xpath('//*[@id="content"]/div/div[2]/div[4]/div[4]/ul/li[3]/span[2]/text()')[0].strip()
        area = text_html.xpath('//*[@id="content"]/div/div[2]/div[4]/div[4]/ul/li[4]/span[2]/text()')[0].strip()
        screen_time = text_html.xpath('//*[@id="content"]/div/div[2]/div[4]/div[4]/ul/li[5]/span[2]/text()')
        film_info = {
            "影名":name,   #strip()方法去掉字符串前后不需要的字符
            "导演": director,
            "主演": main_actors,
            "类型": type0,
            "地区": area,
            "上映时间": screen_time
        }
        films_info.append(film_info)
    except:
        continue  #如果某个链接爬取失败，就跳过，继续爬取下一个
i += 1
time.sleep(2)  #每次请求间隔2秒，防止被封IP
print(f"第{i}个影评已经爬取完成！")

DICT_LIST = films_info  #以后有类似的字典列表，可以直接用这个变量名
keys = list(DICT_LIST[0].keys())  #提取字典的键名作为excel的表头
for i in range(len(keys)):
    sheet1.write(0,i,keys[i])
for col in range(1,len(DICT_LIST)+1):
    for row,key in zip(range(len(keys)),keys):  #zip()函数把两个列表打包成元组列表
        sheet1.write(col,row,DICT_LIST[col-1][key])  

workbook.save(r"C:\Users\Lst12\Desktop\爬虫练习.xls")  #save()方法保存工作簿，注意路径中的反斜杠要用r表示原始字符串，或者用双反斜杠