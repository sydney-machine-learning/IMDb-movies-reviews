import csv
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import time


star_ratings = [1,2,3,4,5,6,7,8,9]  # 对应的星级文件
input_file_template = '250_{star}_star_rated_movies.tsv'
output_file_template = '250_{star}_star_rated_movies_reviews.csv'

for star in star_ratings:
    input_file = input_file_template.format(star=star)
    output_file = output_file_template.format(star=star)

    # 读取TSV文件，获取tconst列的值
    df = pd.read_csv(input_file, sep='\t')  # 随时切换文件名
    tconst_list = df['tconst'].tolist()

    # 创建一个保存所有评论的CSV文件
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['MovieID', 'Review'])  # CSV的列名

        # 启动Chrome浏览器 (确保已安装Chrome浏览器和相应的ChromeDriver)
        driver = webdriver.Chrome()

        # 遍历每个tconst，替换URL并提取评论
        for tconst in tconst_list:
            url = f'https://www.imdb.com/title/{tconst}/reviews'

            # 打开IMDb评论页面
            driver.get(url)

            # 模拟滚动页面以加载更多评论（IMDb评论页面使用懒加载技术，需要滚动）
            for _ in range(5):  # 调整滚动次数以加载更多评论
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # 等待页面加载

            # 获取页面内容并使用BeautifulSoup解析
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 查找所有评论部分
            reviews = soup.find_all('div', class_='text show-more__control')

            # 将每个电影的评论添加到CSV文件中
            for review in reviews:
                writer.writerow([tconst, review.text.strip()])  # 记录电影ID和评论

            print(f"已完成对 {tconst} 的评论提取。")

        # 关闭浏览器
        driver.quit()

    print(f"所有评论已成功保存到 '250_{star}_star_rated_movies_reviews.csv' 文件中。")












