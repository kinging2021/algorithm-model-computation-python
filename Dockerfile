FROM harbor.enn.cn/bigdata/calc_server:0.6

COPY . /calc_server
WORKDIR /calc_server
ENV PATH /root/miniconda3/bin:/root/miniconda3/condabin:$PATH
RUN conda install -y --file requirements_conda.txt
RUN pip install -r requirements_pip.txt

CMD ["gunicorn", "--chdir", "/calc_server", "-c", "/calc_server/conf/gunicorn_conf.py", "run_server:app"]