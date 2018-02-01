FROM hivemall/latest:20170517

COPY ./fix-hivemall-build.patch ${HIVEMALL_PATH}
COPY ./core-site.xml ${HADOOP_HOME}/etc/hadoop
COPY ./init.sh ./bin/

RUN apt update && \
    apt install -y patch && \
    cd ${HIVEMALL_PATH} && \
    patch -p1 < fix-hivemall-build.patch && \
    ./bin/build.sh

EXPOSE 10000

CMD ["sh", "-c", "./bin/init.sh"]
