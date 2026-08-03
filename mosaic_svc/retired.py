class RetiredPipelineError(RuntimeError):
    pass


def reject_r16() -> None:
    raise RetiredPipelineError(
        "Mosaic-SVC R1.6/P11-P16 was retired after an unacceptable subjective audio result. "
        "Use the frozen Seed-VC P0-P10 path instead."
    )
