class CheckpointNotFound(ValueError):
    pass


class ReflectionError(ModuleNotFoundError):
    pass


class FeaturizationError(ValueError):
    pass


class TransformerError(ValueError):
    pass


class SplitError(ValueError):
    pass
