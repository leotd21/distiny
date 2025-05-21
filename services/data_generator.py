
from sdv.tabular import GaussianCopula, CTGAN, TVAE

def generate(df, num_rows, model_type):
    model_cls = {
        "GaussianCopula": GaussianCopula,
        "CTGAN": CTGAN,
        "TVAE": TVAE
    }.get(model_type, GaussianCopula)

    model = model_cls()
    model.fit(df)
    return model.sample(num_rows)
