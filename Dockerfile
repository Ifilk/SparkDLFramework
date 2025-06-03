FROM bitnami/spark:3.5 AS base
LABEL authors="suanx"
USER root
# RUN install_packages openjdk-17-jdk maven
WORKDIR /opt/app

# ---------- 构建阶段 ----------
#FROM base AS build
#COPY . .
# 提前编译(现场编译太tm慢了)
# RUN ./gradlew shadowJar
# ---------- 运行阶段 ----------
#FROM base AS run
WORKDIR /opt/app
# 拷贝提取编译好的Jar包
#COPY --from=build /opt/app/build/libs/*.jar /opt/app/app.jar
COPY spark-submit.sh /opt/app/spark-submit.sh
# 报504，更换 apt 源
RUN sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y netcat-openbsd

RUN chmod +x /opt/app/spark-submit.sh
