import warnings

warnings.warn(
    "insurance-validation is deprecated. Install insurance-governance instead: pip install insurance-governance",
    DeprecationWarning,
    stacklevel=2,
)

from insurance_governance.validation import *  # noqa: F401, F403, E402
