PyHivemall
===

**PyHivemall** enables Python code to access and use machine learning model created by [Apache Hivemall](https://github.com/apache/incubator-hivemall), a scalable machine learning library for Apache Hive.

## Installation

```
$ pip install git+https://github.com/takuti/pyhivemall.git
```

## Usage

### Connect to [HiveServer2](https://cwiki.apache.org/confluence/display/Hive/Setting+Up+HiveServer2)

```
$ hive --service hiveserver2
```

```py
from pyhivemall import HiveConnection

conn = HiveConnection()
```

### Connect to [Treasure Data](https://www.treasuredata.com/)

```py
import os
from pyhivemall import TdConnection

conn = TdConnection(apikey=os.environ['TD_API_KEY'],
                    endpoint=os.environ['TD_API_SERVER'],
                    database='sample_datasets')
```

### Load model

```py
lr = load_model('LogisticRegression', 'lr_model_table', conn,
                feature_column='feature', weight_column='weight', bias_feature='bias')
```