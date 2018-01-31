## Running on Docker

```
$ docker build -t pyhivemall .
$ docker run --rm -it -p 10000:10000 pyhivemall sh -c "./bin/init.sh && hive --service hiveserver2"
```