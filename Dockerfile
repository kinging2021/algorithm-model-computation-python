FROM harbor.enn.cn/bigdata/calc_server:0.2

WORKDIR /
RUN cat /hosts >> /etc/hosts
COPY . /calc_server

WORKDIR /calc_server
RUN pip install -r requirements.txt

ENTRYPOINT ["/app/bin/docker-entrypoint"]
CMD ["/usr/local/bin/gunicorn -c ./conf/gunicorn_conf.py run_server:app"]
