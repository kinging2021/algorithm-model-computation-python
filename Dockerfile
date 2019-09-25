FROM harbor.enn.cn/bigdata/calc_server:0.2

RUN cat /hosts >> /etc/hosts
COPY . /calc_server
WORKDIR /calc_server
RUN pip install -r requirements.txt

CMD ["/usr/local/bin/gunicorn -c /calc_server/conf/gunicorn_conf.py run_server:app"]


