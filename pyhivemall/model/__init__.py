from .logistic_regression import LogisticRegression


def load_model(model, table, conn, **kwargs):
    df = conn.fetch_table(table, columns=[kwargs['feature_column'], kwargs['weight_column']])
    if model == 'LogisticRegression':
        return LogisticRegression.load(source_dataframe=df, **kwargs)
    raise ValueError("model should be 'LogisticRegression', got %s" % model)


__all__ = ['LogisticRegression']
