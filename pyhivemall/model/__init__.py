from .logistic_regression import LogisticRegression


def load_model(model, table, conn, **kwargs):
    df = conn.fetch_table(table)
    if model == 'LogisticRegression':
        return LogisticRegression(source_dataframe=df, **kwargs)
    raise ValueError("model should be 'LogisticRegression', got %s" % model)


__all__ = ['LogisticRegression']
