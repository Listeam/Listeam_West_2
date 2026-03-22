import requests
if __name__ == "__main__":
    url = 'https://www.sogou.com/web'  #搜索引擎进入新网页一般都有参数跟在？后面
    headers ={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
    }
    keywords = input("enter a word:\n")  #输入你给搜索引擎网页的参数，可以动态获得想要页面的html
    param = {              #requests有两个参数。第一个是网页参数值，第二个是UserAgent，都要以字典形式封装
        "query":keywords     
    }
    response = requests.get(url,params = param,headers = headers)
    page_text = response.text

    file_name = keywords + '.html'
    with open(file_name,'w',encoding='utf-8-sig') as file:
        file.write(page_text)  #保存为html类文件

