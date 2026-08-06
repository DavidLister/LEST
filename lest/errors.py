class LestError(Exception):
    """User-facing error; message printed to stderr, exits with `exit_code`."""

    exit_code = 1


class EnvironmentError_(LestError):
    """The environment is not usable (Ollama unreachable, model missing, DB locked)."""

    exit_code = 2
