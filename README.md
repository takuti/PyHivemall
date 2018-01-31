PyHivemall
===

**PyHivemall** enables Python code to access and use machine learning model created by [Apache Hivemall](https://github.com/apache/incubator-hivemall), a scalable machine learning library for Apache Hive.

## Installation

```
$ pip install git+https://github.com/takuti/PyHivemall.git
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

### Connect to [Treasure Data](https://www.treasuredata.com/live-data-platform/)

```py
import os
from pyhivemall import TdConnection

conn = TdConnection(apikey=os.environ['TD_API_KEY'],
                    endpoint=os.environ['TD_API_SERVER'],
                    database='sample_datasets')
```

### Load model with vectorizer

```py
lr, vectorizer = load_model('LogisticRegression', 'lr_model_table', conn,
                            feature_column='feature', weight_column='weight', bias_feature='bias')
```

Note that obtained model is compatible with corresponding [scikit-learn](http://scikit-learn.org/) model; that is, `lr` has the same parameters and functions as the [`LogisticRegression` model in scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

### Vectorize and predict

```py
d = [{'categorical1': 'foo', 'categorical2': 'xxx', 'quantitative': 2.0},
     {'categorical1': 'bar', 'categorical2': 'yyy', 'quantitative': 4.0}]
X = vectorizer.transform(d)
lr.predict(X)  # yields 0/1 binary label
```