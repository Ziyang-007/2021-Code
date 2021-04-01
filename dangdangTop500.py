import requests
from bs4 import BeautifulSoup


def main(page):
    url = 'http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-' + str(page)
    html = request_dangdang(url)
    parse_result(html)

    # for item in items:
    #     write_item_to_file(item)


def request_dangdang(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/89.0.4389.90 Safari/537.36 '
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None


def parse_result(html):
    soup = BeautifulSoup(html, 'lxml')
    items = soup.find('ul', class_="bang_list").find_all('li')

    for item in items:
        rank = item.find(class_='list_num').text[:-1]
        url = item.find('a')['href']
        name = item.find(class_='name').text
        price = item.find(class_='price_n').text
        publisher_total = item.find_all(class_='publisher_info')
        publisher_list = []
        for publisher in publisher_total:
            publisher_list.append(publisher.text)
        # print(rank, url, name, publisher_list[0], publisher_list[1], price)
        result = "rank:" + rank + "; " + "url:" + url + "; " + "name:" + name + "; " + "author:" + publisher_list[
            0] + "; " + "publisher:" + publisher_list[1] + "; " + "price:" + price
        write_item_to_file(result)
        # print(result)


def write_item_to_file(item):
    print('开始写入数据 ====> ' + item)
    with open('book.txt', 'a') as f:
        f.write(item + '\n')


if __name__ == "__main__":
    for i in range(1, 26):
        main(i)
