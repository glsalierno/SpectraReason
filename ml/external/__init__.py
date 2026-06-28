"""External / open FTIR dataset ingestion (experimental tier only)."""

from ml.external.provenance import PREPROCESSING_VERSION, attach_provenance, make_reference_id

__all__ = ["PREPROCESSING_VERSION", "attach_provenance", "make_reference_id", "__version__"]
__version__ = "0.1.0"
