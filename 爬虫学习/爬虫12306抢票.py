import openpyxl #引入打开excel文档的模块
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By #引入元素定位
from selenium.webdriver.support.ui import WebDriverWait  #引入显示等待
from selenium.webdriver.support import expected_conditions as ec   #引入显示等待结束需要的情况
from selenium.webdriver.support.ui import Select #引入专门用来对付下拉列表的模块
from selenium.common.exceptions import NoSuchElementException #引入异常模块，专用于try
def get_page():
    arg = Options()  
    arg.add_experimental_option("detach",True) 
    page = webdriver.Edge(service = Service(r'D:\建模学习listing\msedgedriver.exe'),options = arg) 
    return page

class Automatic_Ticket():
    def __init__(self,from_Station,to_Station,train_date):
        self.from_Station = from_Station
        self.to_Station = to_Station
        self.train_date = train_date
        self.RAILWAY = get_page()
        self.station_code = self.collect_station_code()
    
    def login(self):
        self.RAILWAY.get('https://kyfw.12306.cn/otn/resources/login.html')
        self.RAILWAY.maximize_window()
        self.login_button = self.RAILWAY.find_element(By.XPATH,'/html/body/div[1]/div[2]/div[2]/ul/li[2]/a')
        self.RAILWAY.implicitly_wait(10)
        self.login_button.click()
        WebDriverWait(self.RAILWAY,1000).until(ec.url_to_be('https://kyfw.12306.cn/otn/view/index.html'))
        #WebDriverWait即显示等待，必须触发until后的条件才能继续，ec.url_to_be,即直到网址变为指定网址才继续执行后续代码
        print('登陆成功')

    def collect_station_code(self):
        workbook = openpyxl.load_workbook('车站代码收集.xlsx') #加载生成的站代码表格
        sheet = workbook.active #启用活动表
        collections = [] 
        for row in sheet.rows: #获取sheet的所有行
            collection = []  #每次都重置为空列表
            for cell in row: #获取每行所有单元格
                collection.append(cell.value)  #获取每行两个单元格的值存入collection
            collections.append(collection) #两个元素的小列表一个一个存入collections
        return dict(collections)#按顺序，按照key，value顺序把小列表一个一个变成字典形式，以便查询

    def search_ticket(self):
        self.RAILWAY.get('https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc')
        self.RAILWAY.implicitly_wait(10)
        fromStation_input = self.RAILWAY.find_element(By.ID,'fromStation') #找到隐藏出发站输入框
        toStation_input = self.RAILWAY.find_element(By.ID,'toStation') #找到隐藏终点站输入框 

        fromStation_code = self.station_code[self.from_Station] #求索起始站中文对应的编码
        toStation_code = self.station_code[self.to_Station] #求索终点站中文对应的编码

        self.RAILWAY.execute_script('arguments[0].value=arguments[1]',fromStation_input,fromStation_code) #js语法绕过所有障碍填充，包括隐藏的
        #execute_script('arguments[0].value=arguments[1]',隐藏框名，输入值)
        self.RAILWAY.execute_script('arguments[0].value=arguments[1]',toStation_input,toStation_code)

        self.RAILWAY.find_element(By.CSS_SELECTOR,'#train_date').clear() #先清空默认
        self.RAILWAY.find_element(By.CSS_SELECTOR,'#train_date').send_keys(self.train_date)
        self.RAILWAY.implicitly_wait(10)
        self.RAILWAY.find_element(By.XPATH,'/html/body/div[2]/div[7]/div[9]/form/div[3]/div/a').click() #点击查询
        
    def reserve(self):
        self.RAILWAY.find_element(By.XPATH,'/html/body/div[3]/div[7]/div[13]/table/tbody/tr[1]/td[13]/a').click() #点击预定
        self.RAILWAY.find_element(By.XPATH,'/html/body/div[1]/div[11]/div[3]/div[2]/div[1]/div[2]/ul/li/label').click() #勾选乘车人

        select_element = self.RAILWAY.find_element(By.ID,'seatType_1')
        selects = Select(select_element) #创建Select对象
        selects.select_by_index(0)  #从Select对象选第一个

        self.RAILWAY.find_element(By.XPATH,'/html/body/div[1]/div[11]/div[5]/a[2]').click() #提交订单

        WebDriverWait(self.RAILWAY,1000).until(
            ec.presence_of_element_located((By.CLASS_NAME,'ticket-check'))
        ) #等待检票窗口出现


        WebDriverWait(self.RAILWAY,1000).until(
            ec.element_to_be_clickable((By.ID,'qr_submit_id')) #until只能传一个参数，要输入Id必须额外加一个括号
            )  #显示等待，直到可以确认订单为止

        final_button = self.RAILWAY.find_element(By.ID,'qr_submit_id') #容易点不到这个按键，所以给他额外做一个循环让他一直点到成功为止
        while final_button:
            try:  #试着点击，并重新获得确认按钮元素
                final_button.click()
                final_button = self.RAILWAY.find_element(By.ID,'qr_submit_id')
            except ElementNotVisibileException:   #如果出现某种情况就执行...此错误为存在但不可见，就是还没加载出来没法点击的意思
                #即如果你点击成功，无法再获得到原来的确认按钮，就break，否则继续click
                break  #跳出循环，异常时停止点击

        #self.RAILWAY.find_element(By.XPATH,'')

    def start(self):
        self.login()
        self.search_ticket()
        self.reserve()

home = Automatic_Ticket('福州','福鼎','2025-10-20')
home.start()


