from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

service = Service(ChromeDriverManager().install())  # 自动匹配 Chrome 版本
driver = webdriver.Chrome(service=service)



# 自动下载并返回 ChromeDriver 的路径
driver_path = ChromeDriverManager().install()
print("ChromeDriver 下载路径：", driver_path)
