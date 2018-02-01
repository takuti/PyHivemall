## Running on Docker

> Note: This Docker container is experimental and may not work correctly.

```
$ docker build -t pyhivemall .
$ docker run --rm -it -p 10000:10000 pyhivemall
```

```py
from pyhivemall import HiveConnection

conn = HiveConnection(host='localhost', port=10000)
```