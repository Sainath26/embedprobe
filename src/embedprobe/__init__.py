"""embedprobe — a diagnostic toolkit for evaluating language-model embedding spaces."""

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy imports keep `import embedprobe` cheap and avoid circular imports.
    if name == "probe":
        from embedprobe.probe import probe

        return probe
    if name in ("ProbeReport", "ModelDiagnostics"):
        from embedprobe import report

        return getattr(report, name)
    raise AttributeError(f"module 'embedprobe' has no attribute {name!r}")


__all__ = ["probe", "ProbeReport", "ModelDiagnostics", "__version__"]
