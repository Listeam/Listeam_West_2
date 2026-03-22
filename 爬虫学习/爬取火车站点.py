import re
import requests
import openpyxl
headers = {
    'UserAgent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
}
stations_list = requests.get('https://kyfw.12306.cn/otn/resources/js/framework/station_name.js',headers = headers)
stations_text = stations_list.text
stations = re.findall(r'([\u4e00-\u9fa5]+)\|([A-Z]+)',stations_text)
#前面一串是中文专用编码，()把要提取的内容括起来表示分组，并且屏蔽|
#print(stations)
workbook = openpyxl.Workbook()
sheet = workbook.active #使用活动表
for station in stations:
    sheet.append(station)  #openpyxl里sheet的append功能会自动换行输入
workbook.save('车站代码收集.xlsx')
