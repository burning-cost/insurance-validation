# insurance-validation — Deprecated

This package has been superseded by [insurance-governance](https://github.com/burning-cost/insurance-governance).

All functionality from insurance-validation — `ModelCard`, `DataQualityReport`, `PerformanceReport`, `DiscriminationReport`, `StabilityReport`, and `ReportGenerator` — is now part of insurance-governance under the `insurance_governance.validation` subpackage. The governance version also includes a numpy 2.0 compatibility fix (`np.trapezoid` / `np.trapz` fallback) and additional methods (`gini_with_ci`, `ae_with_poisson_ci`, `hosmer_lemeshow_test`, `double_lift`).

## Migration

```bash
pip install insurance-governance
```

```python
# Before
from insurance_validation import (
    ModelCard, DataQualityReport, PerformanceReport,
    DiscriminationReport, StabilityReport, ReportGenerator,
)

# After
from insurance_governance.validation import (
    ModelCard, DataQualityReport, PerformanceReport,
    DiscriminationReport, StabilityReport, ReportGenerator,
)
```

This repository is archived and will not receive further updates.
