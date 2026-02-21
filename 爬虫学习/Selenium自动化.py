from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By #导入元素定位模块
import time

def get_page():
    arg = Options()  #创建设置浏览器可选参数对象
    #arg.add_argument(" --no-sandbox") 
    arg.add_experimental_option("detach",True) #保持浏览器打开状态，默认是代码运行完就关闭，experimental_option()方法设置实验性参数，第一个参数是默认分离表示浏览器网页和代码进程分离，第二个参数是是否保持打开
    page = webdriver.Edge(service = Service(r'D:\VScode\msedgedriver.exe'),options = arg) #创建浏览器对象，指定浏览器驱动路径，并附上先前创建的参数对象
    return page
p1 = get_page()
#打开指定网址
p1.get('https://www.baidu.com/')
p1.implicitly_wait(4)  #隐式等待，最长等10秒，网页加载完就执行后续代码比直接sleep(4)更高效
#time.sleep(4)
#p1.close() #关闭当前网页
#p1.quit() #关闭所有网页

# p1.maximize_window() #最大化
# time.sleep(2)
# p1.minimize_window() #最小化
# time.sleep(2)

p1.set_window_position(300,50)  #设置浏览器位置
p1.set_window_size(800,600)  #设置浏览器大小
#p1.get_screenshot_as_file('doubao.png')  #截图并保存
time.sleep(2)
#p1.refresh()#刷新网页

p1.maximize_window()
time.sleep(1)
#a1 = p1.find_element(By.ID,'chat-textarea')#调用By模块的查找Id功能，第二个参数即ID 

a1 = p1.find_element(By.CSS_SELECTOR,'#chat-textarea')  #CSS选择器定位，#加id值表示id，.加class值表示class
#CSS_SELECTOR不加修饰符就是标签头，如div,span等，"[类型='精准值']"表示任意属性即除了常见属性外的属性，如data-开头的自定义属性
#类型后加^表示属性值以什么开头，$表示属性值以什么结尾，*表示属性值包含什么，类似partial的模糊查询
#最简便的是直接复制selector路径，粘贴即可，也可以用xpath功能完整路径定位避免网页随机变化导致定位失败
#a1 = p1.find_elements(By.ID,'chat-route-layout') #find_elements返回的是列表  

a2 = p1.find_element(By.ID,'chat-submit-button') #还可以为name，class_name，这两个可以在网页控制台里搜document.getElementsByClassName('class值')或xpath等定位方式判断是列表还是单独元素，以判断是否要切片
a1.send_keys('豆包字节跳动官方') #元素输入，在网页对应位置输入，也可以输入图片相对路径
p1.implicitly_wait(4) 
a2.click() #元素点击
a3 = p1.find_element(By.LINK_TEXT,'地图') #通过链接文本定位，partial_link_text是部分的模糊文本定位，也能获得    
a3.click()
p1.implicitly_wait(4) 
a4 = p1.find_element(By.CLASS_NAME,'searchbox-content-common')
b4 = p1.find_element(By.ID,'search-button')
#遇到自动重叠的网页就可以直接检查-查询到元素，但如果是自动开新网页就需要切换窗口句柄
#p_nums = p1.window_handles #获取在原网页基础上打开的所有窗口句柄，返回列表
#p1.close()  #关闭先前网页
#p1.switch_to.window(p_nums[1]) #从p1切换到输入的句柄对应网页的window

#p2 = p1.current_window_handle  #获取当前窗口句柄
#p1.switch_to.window(p2) #切换到当前窗口句柄对应网页
a4.send_keys('福鼎市')
p1.implicitly_wait(4) 
b4.click()
#a1.clear() #元素清空

p1.back()  #浏览器后退
time.sleep(1)
p1.forward()  #浏览器前进

#设p3为新打开的有警戒弹窗窗口
#a5 = p1.switch_to.alert.text #切换到警戒弹窗并获取文本
#p3.switch_to.alert.accept()#切换到警戒弹窗并接受
#p3.switch_to.alert.dismiss()#切换到警戒弹窗并取消
#p3.switch_to.alert.send_keys() #切换到警戒弹窗并输入文本

#当需要操作的元素归属于iframe时，需要先切换到iframe
#a6 = p1.find_element(By.Xpath,'iframe的地址')#定位iframe值
#p1.switch_to.frame(a6) #用获得的iframe值切换到frame
#然后就可以正常在新frame找需要操作的元素再进行操作
#p1.switch_to.default_content() #操作完毕后切换回主frame


time.sleep(6)
p1.close() #关闭当前网页