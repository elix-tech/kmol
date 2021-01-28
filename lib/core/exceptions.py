class CheckpointNotFound(ValueError):
    pass


class ReflectionError(ModuleNotFoundError):
    pass


class FeaturizationError(ValueError):
    pass


class SplitError(ValueError):
    pass
