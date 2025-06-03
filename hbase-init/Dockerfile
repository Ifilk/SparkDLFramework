FROM robd003/python3.10:latest
LABEL authors="suanx"
WORKDIR /opt/app
COPY . .

# 安装依赖
# 报504，更换 apt 源
RUN sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list
# thriftpy2 编译需要gcc
RUN apt-get update && apt-get install -y gcc netcat-openbsd
RUN pip3 install cython thriftpy2
RUN pip3 install -r ./requirements.txt