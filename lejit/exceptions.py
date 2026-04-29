class LeJITError(Exception):
    pass


class ConfigurationError(LeJITError):
    pass


class ConstraintViolationError(LeJITError):
    pass


class UnsupportedPromptError(LeJITError):
    pass
