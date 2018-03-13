## Running on Docker

> Container orchestration is based on [big-data-europe/docker-hive](https://github.com/big-data-europe/docker-hive)

First and foremost, get source code and build [Hivemall v0.5.0-rc.3](https://github.com/apache/incubator-hivemall/releases/tag/v0.5.0-rc3), and place `target/hivemall-all-0.5.0-incubating.jar` in here.

Next, run containers:

```
$ docker-compose up -d
```

(Wait for a while until HiveServer2 is successfully launched.)

Now, the [breast cancer wisconsin dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) is automatically stored to `default.breast_cancer` table:

```
$ docker-compose exec hive-server bash
# /opt/hive/bin/beeline -u jdbc:hive2://localhost:10000
> show tables;
+----------------+
|    tab_name    |
+----------------+
| breast_cancer  |
+----------------+
> select count(1) from breast_cancer;
+------+
| _c0  |
+------+
| 569  |
+------+
```

Let's build a Logistic Regression model on the data:

```
> !run /root/logistic_regression.sql
> select feature, weight from breast_cancer_logress_model order by abs(weight) desc limit 5;
+----------+---------------------+
| feature  |       weight        |
+----------+---------------------+
| f23      | 1129.2266845703125  |
| f3       | 1099.8594970703125  |
| f24      | -663.974853515625   |
| f22      | 353.6787109375      |
| f4       | 346.53466796875     |
+----------+---------------------+
```

Eventually, you can access to the logistic regression model on your local environment:

```py
from pyhivemall import HiveConnection
from pyhivemall.linear_model import SGDClassifier

conn = HiveConnection(host='localhost', port=10000)
clf, vectorizer = SGDClassifier.load(conn, 'breast_cancer_logress_model',
                                     feature_column='feature',
                                     weight_column='weight',
                                     bias_feature='0',
                                     options='-loss log -opt SGD -reg l1 -eta fixed')
```

Check the accuracy of prediction in Python:

```py
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import recall_score, precision_score, f1_score

breast_cancer = load_breast_cancer()

# create input for the vectorizer with corresponding feature name
d = list()
feature_names = ['f%d' % i for i in range(1, len(breast_cancer.feature_names) + 1)]
for sample in breast_cancer.data:
    d.append(dict(zip(feature_names, sample)))

X = vectorizer.transform(d)

y_true, y_pred = breast_cancer.target, clf.predict(X)
recall_score(y_true, y_pred)     # => 0.37254901960784315
precision_score(y_true, y_pred)  # => 1.0
f1_score(y_true, y_pred)         # => 0.5428571428571428
```

Insufficient accuracy? Try to re-fit the model by using the true samples:

```py
clf.fit(X, y_true)

y_pred = clf.predict(X)
recall_score(y_true, y_pred)     # => 0.988795518207283
precision_score(y_true, y_pred)  # => 0.8936708860759494
f1_score(y_true, y_pred)         # => 0.9388297872340425
```

(Of course it's cheating in reality, though :P)

Eventually, it's time to store the new model to Hive:

```py
clf.store(conn, 'breast_cancer_logress_model_sklearn', vectorizer.vocabulary_, bias_feature='0')
```

Check the difference between two models built by Hivemall and scikit-learn:

```
> select t1.feature, t1.weight as weight_hivemall, t2.weight as weight_sklearn from breast_cancer_logress_model t1 join breast_cancer_logress_model_sklearn t2 on t1.feature = t2.feature limit 5;
+-------------+---------------------+-----------------+
| t1.feature  |   weight_hivemall   | weight_sklearn  |
+-------------+---------------------+-----------------+
| f5          | 1.827925443649292   | -1.73197169E10  |
| f28         | -1.39395272731781   | 2.77178352E11   |
| f9          | 3.4602367877960205  | -2.57569653E11  |
| f23         | 1129.2266845703125  | 1.85949288E13   |
| f13         | 2.792898654937744   | -8.0768412E12   |
+-------------+---------------------+-----------------+
```