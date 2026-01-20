from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import timedelta, datetime
import concurrent.futures
import traceback
import pandas as pd
import requests


def crawler_flight(flight_num):
    day = datetime.now()
    times = [None, None, None, None]
    for _ in range(1):
        day = day + timedelta(days = 1)
        day_str = day.strftime("year=%Y&month=%m&date=%d")
        station_data_url = f'https://www.flightstats.com/v2/flight-details/{str(flight_num)}?{day_str}'
        response = requests.get(station_data_url)
        if response.status_code == 200:
            try:
                page_content = response.text
                soup = BeautifulSoup(page_content, 'html.parser')
                nodes = soup.select("div.innerStyle div.timeBlock:has(> p.title:-soup-contains('Scheduled')) h4")
                times = [n.get_text(strip=True) for n in nodes]
                if len(times) == 4 and times != [None, None, None, None]:
                    return times
            except:
                return [None, None, None, None]
    return [None, None, None, None]


def crawler_flights(flight_nums):
    flight_nums = list(set(flight_nums))
    flight_nums = [_[0:2] + '/' + _[2:] for _ in flight_nums if len(_) == 6]
    res = []
    with tqdm(total=len(flight_nums), desc=f'Crawl flight', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        for _ in flight_nums:
            time_ = crawler_flight(_)
            no_time = [_]
            no_time.extend(time_)
            res.append(no_time)
            print(no_time)
            pbar.update()
    a = pd.DataFrame(res, columns=['flight_no', 'scheduled_depart','actual_depart', 'scheduled_arrival', 'actual_arrival'])
    a.to_csv('fligt.csv')


if __name__ == '__main__':
    try:
        df = pd.read_excel('flight_no.xlsx', engine='openpyxl')
        flight_no = df['NO'].tolist()
        crawler_flights(flight_no)
    except Exception as e:
        print("\n程序出错了：", e)
        traceback.print_exc()  # 打印完整错误堆栈
        input("\n按回车键退出...")