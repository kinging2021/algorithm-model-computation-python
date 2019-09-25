FROM harbor.enn.cn/bigdata/calc_server:0.2

RUN cat /hosts >> /etc/hosts
COPY . /calc_server
WORKDIR /calc_server
RUN pip install -r requirements.txt

ENTRYPOINT ["/usr/local/bin/gunicorn -c /calc_server/conf/gunicorn_conf.py /calc_server/run_server:app"]


