FROM harbor.enn.cn/bigdata/calc_server:0.6

COPY . /calc_server
WORKDIR /calc_server
RUN conda install -y --file requirements_conda.txt
RUN pip install -i -r requirements.txt

CMD ["/usr/local/bin/gunicorn", "--chdir", "/calc_server", "-c", "/calc_server/conf/gunicorn_conf.py", "run_server:app"]