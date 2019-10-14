FROM fnwharbor.enncloud.cn/fnw/ubuntu-with-python:python-3.6

COPY . /calc_server
WORKDIR /calc_server
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

CMD ["/usr/local/bin/gunicorn", "--chdir", "/calc_server", "-c", "/calc_server/conf/gunicorn_conf.py", "run_server:app"]