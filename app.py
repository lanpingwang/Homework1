# -*- coding: utf-8 -*-


from __future__ import unicode_literals
# 引入套件
from flask import Flask, request, abort
# 初始化Flask物件
app = Flask(__name__)

# 這裡定義服務存放的位置，也就是使用者連線的網址


@app.route("/", methods=['GET'])
# 這個服務對應的方法
def basic_url():
    # return 來回傳使用者要收到的資料
    return 


@app.route("/map/w01-6", methods=['GET'])
def map_w01_6():
    return app.send_static_file('index.html')


if __name__ == "__main__":
    app.run()
