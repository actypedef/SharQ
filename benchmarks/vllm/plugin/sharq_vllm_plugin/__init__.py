"""vLLM general plugin that registers the `sharq` quantization method."""


def register() -> None:
    from . import sharq  # noqa: F401  (registers SHARQConfig + custom op on import)
