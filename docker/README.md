## Running on Docker

> Container orchestration is based on [big-data-europe/docker-hive](https://github.com/big-data-europe/docker-hive)

Get source code and build [Hivemall v0.4.2-rc.2](https://github.com/apache/incubator-hivemall/releases/tag/v0.4.2-rc.2), and place `target/hivemall-core-0.4.2-rc.2-with-dependencies.jar` under this directory.

Run containers:

```
$ docker-compose up -d
```

(Wait for a while until HiveServer2 is successfully launched.)

Now, the [breast cancer wisconsin dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) is automatically stored to `default.cancer` table:

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
> !run /root/logistic_regression.sql
> select feature, weight from breast_cancer_logress_model order by abs(weight) desc limit 5;
+----------+----------------------+
| feature  |        weight        |
+----------+----------------------+
| f24      | -190.29006958007812  |
| f4       | 180.16073608398438   |
| f3       | 161.74786376953125   |
| f23      | 155.4984893798828    |
| f22      | 60.91228103637695    |
+----------+----------------------+
```

Eventually, you can access to the logistic regression model on your local environment:

```py
from pyhivemall import HiveConnection

conn = HiveConnection(host='localhost', port=10000)
lr, vectorizer = load_model('LogisticRegression', 'breast_cancer_logress_model', conn,
                            feature_column='feature', weight_column='weight', bias_feature='0')
```