# Python爬虫实例——当当网Top500图书排行



## 前期准备

1. 使用chrome浏览器进入网址http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-1 ，按 **f12** 进入开发者工具查看网页源代码 ~~（chrome永远滴神）~~。


2. 点击上方导航栏里的**Network**，可以查看到当前HTTP消息的状态，还可以看到当前网页的请求头**Resquest Headers**。

<img src="https://github.com/crazySao/2021-Code/blob/main/spider/当当网Top500书籍排名/image/image-20210330151532432.png?raw=true" alt="image-20210330151532432"  />

> - HTTP状态码，当浏览者访问一个网页时，浏览者的浏览器会向网页所在服务器发出请求。当浏览器接收并显示网页前，此网页所在的服务器会返回一个包含HTTP状态码的信息头（server header）用以响应浏览器的请求。HTTP状态码的英文为HTTP Status Code。状态代码由三位数字组成，第一个数字定义了响应的类别，且有五种可能取值。 
>    - 1xx：指示信息–表示请求已接收，继续处理。
>    - 2xx：成功–表示请求已被成功接收、理解、接受。
>    - 3xx：重定向–要完成请求必须进行更进一步的操作。
>    - 4xx：客户端错误–请求有语法错误或请求无法实现。
>    - 5xx：服务器端错误–服务器未能实现合法的请求。
> - 常见状态代码、状态描述的说明如下。 
>   - 200 OK：客户端请求成功。
>   - 400 Bad Request：客户端请求有语法错误，不能被服务器所理解。
>   - 401 Unauthorized：请求未经授权，这个状态代码必须和WWW-Authenticate报头域一起使用。
>   - 403 Forbidden：服务器收到请求，但是拒绝提供服务。
>   - 404 Not Found：请求资源不存在，举个例子：输入了错误的URL。
>   - 500 Internal Server Error：服务器发生不可预期的错误。
>   - 503 Server Unavailable：服务器当前不能处理客户端的请求，一段时间后可能恢复正常，举个例子：HTTP/1.1 200 OK（CRLF）。

> - 其中HTTP有多种请求方式，后续的爬虫编程中常使用**get**方式，获取网页数据。

<img src="https://img-blog.csdn.net/20170326230025261?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamF2YW5kcm9pZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center"  >



3. 在**Respond**中可以看见服务器返回的数据，也是当前网页的html源代码，我们仔细查看需要爬去的信息部分的html代码，可以发现每一个排名的图书信息都被放在了 **< ul >** 标签下的 **< li >** 这个标签里面，包括图书的**排名**、**书名**、**作者**等，那这个部分就是我们需要爬去的数据。

<img src="https://github.com/crazySao/2021-Code/blob/main/spider/当当网Top500书籍排名/image/image-20210330162156995.png?raw=true" alt="image-20210330162156995"  />   



4. 除此之外，还注意到一页显示的排名是20本书，并且点击下一页后，网址变成了http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-2 ，那么我们等会在 python 中可以用一个变量来实现获取不同页数的内容。

----



   ## Python编程

1. 引入所需要的头文件

   ```python
   from bs4 import BeautifulSoup
   import requests
   ```

> - 其中requests是python实现的最简单易用的HTTP库，用了进行网页请求，获取网页的响应；re是正则表达式的库；BeautifulSoup是用于解析html代码，便于获取需要的文本。



2. requests请求的函数

   ```python
   def request_dangdang(url):
       headers = {
           'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 '
       }
       try:
           response = requests.get(url, headers=headers)
           if response.status_code == 200:
               return response.text
       except requests.RequestException:
           return None
   ```
   
> - 使用requests.get方法获取网址的响应，其中可以添加在chrome中获取到的请求头**Request Head**中内的内容，里面有浏览器属性等信息，可以让你的请求更像是浏览器发出的，是其中一种对付反爬虫的手法    ~~(低级的反爬虫)~~。



3. 使用BeautifulSoup解析html的函数，并将数据写入文本中

   ```python
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
           result = "rank:" + rank + "; " + "url:" + url + "; " + "name:" + name + "; " + "author:" + publisher_list[0] + "; " + "publisher:" + publisher_list[1] + "; " + "price:" + price
           # print(result)
           write_item_to_file(result)
   ```

> - BeautifulSoup用来解析 HTML 比较简单，API非常人性化，支持CSS选择器、Python标准库中的HTML解析器，也支持 lxml 的 XML解析器。
>   | **解析工具**  | **解析速度** | **使用难度** |
>   | :-----------: | :----------: | :----------: |
>   | BeautifulSoup |     最慢     |    最简单    |
>   |     lxml      |      快      |     简单     |
>   |  正则表达式   |     最快     |     最难     |
>
>
> - 搜索文档树，一般用得比较多的就是两个方法，一个是**find**，一个是**find_all**。find方法是找到第一个满足条件的标签后就立即返回，**只返回一个元素**。find_all方法是把**所有满足条件**的标签都选到，然后返回回去。使用这两个方法，最常用的用法是出入name以及attr参数找出符合要求的标签。
> - 通过 text 参数可以搜搜文档中的字符串内容.与 name 参数的可选值一样, text 参数接受字符串, 正则表达式 , 列表, True 。
> ```   python
> ###来自参考文献的例子
> soup.find_all(text="Elsie")
> #[u'Elsie']
> 
> soup.find_all(text=["Tillie", "Elsie", "Lacie"])
> #[u'Elsie', u'Lacie', u'Tillie']
> 
> soup.find_all(text=re.compile("Dormouse"))
> [u"The Dormouse's story", u"The Dormouse's story"]
> 
> def is_the_only_string_within_a_tag(s):
>     ""Return True if this string is the only child of its parent tag.""
>     return (s == s.parent.string)
> 
> soup.find_all(text=is_the_only_string_within_a_tag)
> #[u"The Dormouse's story", u"The Dormouse's story", u'Elsie', u'Lacie', u'Tillie', u'...']
> ```
>
> - **Python 列表(List)**
>
>   ```python
>   list = []         ## 空列表 
>   list.append('Google')   ## 使用 append() 添加元素 
>   list.append('Runoob') 
>   print (list)
>   ```
>
>   输出结果：
>
>   ```python
>   ['Google', 'Runoob']
>   ```


4. 数据写入文档的函数

   ```python
   def write_item_to_file(item):
       print('开始写入数据 ====> ' + item)
       with open('book.txt', 'a') as f:
           f.write(item + '\n')
   ```
> - 这里open()里的参数要使用 **a** 追加，不能用 **w** 否则会覆盖之前循环写入的数据
>
> - **with open() as f 用法**，常见的读写操作：
>
>   ```python
>   with open(r'filename.txt') as f:
>      data_user=pd.read_csv(f)  #文件的读操作
>   
>   with open('data.txt', 'w') as f:
>      f.write('hello world')  #文件的写操作
>   ```
>
>   相关参数：
>
>   - r:    以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。
>   - rb: 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
>   - r+: 打开一个文件用于读写。文件指针将会放在文件的开头。
>   - rb+:以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。
>   - w:    打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
>   - wb:    以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
>   - w+:    打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
>   - wb+:以二进制格式打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
>   - a:    打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
>   - ab:    以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
>   - a+:    打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
>   - ab+:以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。
>
>   file对象的属性：
>
>   - file.read([size])   将文件数据作为字符串返回，可选参数size控制读取的字节数
>   - file.readlines([size])   返回文件中行内容的列表，size参数可选
>   - file.write(str)   将字符串写入文件
>   - file.writelines(strings)   将字符串序列写入文件
>   - file.close()   关闭文件
>   - file.closed    表示文件已经被关闭，否则为False
>   - file.mode    Access文件打开时使用的访问模式
>   - file.encoding    文件所使用的编码
>   - file.name    文件名
>   - file.newlines    未读取到行分隔符时为None，只有一种行分隔符时为一个字符串，当文件有多种类型的行结束符时，则为一个包含所有当前所遇到的行结束的列表
>   - file.softspace    为0表示在输出一数据后，要加上一个空格符，1表示不加。这个属性一般程序员用不着，由程序内部使用




5. 代码的主函数

   ```python
   def main(page):
       url = 'http://bang.dangdang.com/books/fivestars/01.00.00.00.00.00-recent30-0-0-1-' + str(page)
       html = request_dangdang(url)
       parse_result(html)
   ```

> - 分别由三个函数组成，分别是**request_dangdang(url)**、**parse_result(html)**，负责对**网页的请求**、**对html代码的解析**以及**将数据写入文本之中**。其中，url的结尾替换为字符串str(page)，用于获取后一页的数据。



6. 循环运行主函数

   ```python
   if __name__ == "__main__":
       for i in range(1, 26):
           main(i)
   ```

---




## 参考文献
1. python爬虫06 | 你的第一个爬虫，爬取当当网 Top 500 本五星好评书籍:https://mp.weixin.qq.com/s?__biz=MzU2ODYzNTkwMg==&mid=2247484142&idx=1&sn=d4893c734e44a16db871f7904910bdcb&chksm=fc8bba7fcbfc336964aa4ee74d490098024479e663b9a83e7ca8e7b4ec876009a9f497462c77&cur_album_id=1321044729160859650&scene=189#rd
2. Http请求头和响应头:https://blog.csdn.net/android_zhengyongbo/article/details/75452305
3. Python爬虫之BeautifulSoup4:https://blog.csdn.net/weixin_45413603/article/details/109799501
4. 【Python爬虫】Beautifulsoup4中find_all函数:https://blog.csdn.net/chengyikang20/article/details/89484033
5. Python列表(List):https://www.runoob.com/python/python-lists.html
6. with open() as f 用法:https://blog.csdn.net/wzhrsh/article/details/101629075

