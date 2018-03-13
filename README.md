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

conn = HiveConnection(host='localhost', port=10000)
```

### Connect to [Treasure Data](https://docs.treasuredata.com/)

```py
import os
from pyhivemall import TdConnection

conn = TdConnection(apikey=os.environ['TD_API_KEY'],
                    endpoint=os.environ['TD_API_SERVER'],
                    database='sample_datasets')
```

### Load model with vectorizer

```py
from pyhivemall.linear_model import SGDClassifier
clf, vectorizer = SGDClassifier.load(conn, 'lr_model_table',
                                     feature_column='feature',
                                     weight_column='weight',
                                     bias_feature='bias')
```

Note that obtained model is basically compatible with corresponding [scikit-learn](http://scikit-learn.org/) model; that is, `clf` has the same parameters and functions as the [`SGDClassifier` model in scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).

### Vectorize and predict

Your vectorizer may have the following features:

```py
vectorizer.feature_names_
# ['categorical1#foo',
#  'categorical1#bar',
#  'categorical1#baz',
#  'categorical2#xxx',
#  'categorical2#yyy',
#  'categorical2#zzz',
#  'quantitative']
```

In that case, prediction can be done by:

```py
d = [{'categorical1': 'foo', 'categorical2': 'xxx', 'quantitative': 2.0},
     {'categorical1': 'bar', 'categorical2': 'yyy', 'quantitative': 4.0}]
X = vectorizer.transform(d)
clf.predict(X)  # yields 0/1 binary label
```

### Rebuild and update model

Of course, re-fitting model in your local environment and storing the new model is possible:

```py
clf.fit(X_train, y_train)
clf.store(conn, 'lr_model_table_sklearn', vectorizer.vocabulary_, bias_feature='bias')
```

Also see an [example running on Docker containers](docker/) for the usage.