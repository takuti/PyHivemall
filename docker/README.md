## Running on Docker

> Container orchestration is based on [big-data-europe/docker-hive](https://github.com/big-data-europe/docker-hive)

```sh
$ wget https://github.com/myui/hivemall/releases/download/v0.4.2-rc.2/hivemall-core-0.4.2-rc.2-with-dependencies.jar
```

```
$ docker-compose up -d
```

```py
from pyhivemall import HiveConnection

conn = HiveConnection(host='localhost', port=10000)
```