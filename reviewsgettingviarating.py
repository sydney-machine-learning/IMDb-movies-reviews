import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
# 读取TSV文件中的tconst列
tconst_list = []
tsv_file = '250_1_star_rated_movies.tsv'  # 请确保文件路径正确
try:
    with open(tsv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')  # 以制表符 \t 作为分隔符
        next(reader)  # 跳过表头
        for row in reader:
            if row:  # 确保不是空行
                tconst_list.append(row[0])  # 第一列是电影ID

    print(f"成功读取 {len(tconst_list)} 部电影的ID。")
except Exception as e:
    print(f"读取文件失败: {str(e)}")

# 设置 ChromeDriver 路径
chrome_driver_path = r"C:\Users\87385\.wdm\drivers\chromedriver\win64\133.0.6943.53\chromedriver-win32\chromedriver.exe"

# 配置 ChromeOptions
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# 启动 WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# 创建一个 CSV 文件来存储评论和评分
with open('imdb_reviews_by_rating1.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['MovieID', 'Rating', 'Review'])  # 添加 Rating 列
    for tconst in tconst_list:
        url = f'https://www.imdb.com/title/{tconst}/reviews'
        driver.get(url)
        print(f"正在处理电影 {tconst}")

        # 等待页面加载
        time.sleep(3)

        # 滚动页面加载更多评论
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # 获取页面内容并解析
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 找到所有评论容器
        review_containers = soup.find_all('div', class_='review-container')

        if not review_containers:
            print("尝试使用新的选择器...")
            # 使用评论内容的父级容器来定位
            review_containers = soup.find_all('div', class_='ipc-html-content-inner-div')

        if not review_containers:
            print(f"警告: {tconst} 没有找到评论，可能 HTML 结构已更改！")
        else:
            reviews_count = 0
            for container in review_containers:
                try:
                    # 获取评论文本
                    review_text = container.text.strip()

                    # 获取评分 - 先查找评论容器的父元素，然后在其中查找评分
                    parent_div = container.parent
                    rating = 'No Rating'
                    while parent_div and parent_div.name == 'div':
                        rating_span = parent_div.find('span', class_='ipc-rating-star--rating')
                        if rating_span:
                            rating = rating_span.text.strip()
                            break
                        parent_div = parent_div.parent

                    # 写入CSV
                    if review_text:
                        writer.writerow([tconst, rating, review_text])
                        reviews_count += 1
                        print(f"已提取第 {reviews_count} 条评论，评分: {rating}")

                except Exception as e:
                    print(f"处理评论时出错: {str(e)}")

            print(f"已完成 {tconst} 的评论提取，获取 {reviews_count} 条评论")
# 关闭浏览器
driver.quit()
print("爬取完成，请检查 imdb_reviews_by_rating1.csv 文件")
