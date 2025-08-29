import csv
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 电影 ID / Movie ID
tconst = 'tt14513804'

# ChromeDriver 路径 / Path to your local chromedriver
chrome_driver_path = r"E:\chromedriver-win64\chromedriver.exe"

# 浏览器设置 / Chrome Options
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# 启动 Chrome 浏览器 / Launch browser
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# IMDb 评论页链接 / IMDb review page
url = f'https://www.imdb.com/title/{tconst}/reviews'
driver.get(url)
print(f"🎬 正在处理电影 {tconst} / Processing movie {tconst}")

# 持续点击“加载更多”按钮直到按钮不再出现或点击失败 / Keep clicking "Load More"
max_clicks = 1000
click_count = 0

while click_count < max_clicks:
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ipc-see-more__text"))
        )
        load_more_button = driver.find_element(By.CLASS_NAME, "ipc-see-more__text")

        if "more" in load_more_button.text.lower():
            driver.execute_script("arguments[0].click();", load_more_button)
            print(f"✅ 第 {click_count + 1} 次点击“加载更多” / Clicked Load More ({click_count + 1})")
            time.sleep(3)  # 等待评论加载
            click_count += 1
        else:
            print("🛑 未检测到“加载更多”按钮文本 / Button text not matching.")
            break

    except (NoSuchElementException, TimeoutException, StaleElementReferenceException, ElementClickInterceptedException) as e:
        print(f"⚠️ 加载更多按钮出错，跳出循环: {str(e)} / Exception: {str(e)}")
        break

# 解析 HTML / Parse HTML content
time.sleep(2)
soup = BeautifulSoup(driver.page_source, 'html.parser')

# 评论容器 / Review containers
review_containers = (
    soup.find_all('div', class_='review-container') or
    soup.find_all('div', class_='ipc-html-content-inner-div')
)

if not review_containers:
    print("⚠️ 使用备用选择器尝试抓取评论 / Trying alternative selector...")
    review_containers = soup.find_all('div', class_='ipc-html-content-inner-div')

if not review_containers:
    print(f"❌ 没有找到评论，可能页面结构已变 / No reviews found for {tconst}")
else:
    # 创建 CSV 文件 / Create CSV file
    output_filename = f'imdb_reviews_{tconst}.csv'
    output_path = os.path.join(os.getcwd(), output_filename)

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['MovieID', 'Rating', 'Review'])

        reviews_count = 0
        for container in review_containers:
            try:
                review_text = container.get_text(strip=True)

                # 向上查找评分 / Traverse parent divs to find rating
                parent_div = container.parent
                rating = 'No Rating'
                while parent_div and parent_div.name == 'div':
                    rating_span = parent_div.find('span', class_='ipc-rating-star--rating')
                    if rating_span:
                        rating = rating_span.get_text(strip=True)
                        break
                    parent_div = parent_div.parent

                if review_text:
                    writer.writerow([tconst, rating, review_text])
                    reviews_count += 1
                    print(f"📄 第 {reviews_count} 条评论，评分: {rating}")
            except Exception as e:
                print(f"⚠️ 处理评论出错: {e}")

    print(f"\n🎉 共提取 {reviews_count} 条评论，保存为：{output_path}")

# 关闭浏览器 / Close browser
driver.quit()
print("✅ 浏览器已关闭，任务完成 / Task completed!")


