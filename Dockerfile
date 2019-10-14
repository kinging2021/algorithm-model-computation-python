FROM fnwharbor.enncloud.cn/bigdata/calc_server:0.2

COPY . /calc_server
WORKDIR /calc_server
RUN pip install -r requirements.txt

#CMD ["/usr/local/bin/gunicorn", "--chdir", "/calc_server", "-c", "/calc_server/conf/gunicorn_conf.py", "run_server:app"]
CMD ["python", "run_server.py"]