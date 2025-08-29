import joblib


class ModelMetadata:
    """
    Wrapper que combina o pipeline com metadados importantes
    """

    def __init__(self, pipeline, target_column, classification_report=None):
        self.pipeline = pipeline
        self.target_column = target_column
        self.classification_report = classification_report

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def fit(self, X, y):
        return self.pipeline.fit(X, y)

    def transform(self, X):
        return self.pipeline.transform(X)

    def __getattr__(self, name):
        # Verificar se o pipeline existe para evitar recursão infinita
        if 'pipeline' not in self.__dict__:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        # Verificar se o pipeline tem o atributo antes de tentar acessá-lo
        if hasattr(self.pipeline, name):
            return getattr(self.pipeline, name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def named_steps(self):
        return self.pipeline.named_steps
