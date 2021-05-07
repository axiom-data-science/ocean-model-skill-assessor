import numpy as np


def _align(reference, sample):
    """Aligns reference and sample signals in time and returns a combined DataFrame"""
    # TODO: Handle gaps and reference signals without a regular frequency
    reference.rename(columns={reference.columns[0]: 'reference'}, inplace=True)
    sample.rename(columns={sample.columns[0]: 'sample'}, inplace=True)
    aligned = reference.join(sample).interpolate()
    return aligned


def compute_bias(reference, sample):
    """Given reference and sample signals return bias."""
    aligned_signals = _align(reference, sample)
    return (aligned_signals['sample'] - aligned_signals['reference']).mean()


def compute_correlation_coefficient(reference, sample):
    """Given reference and sample signals, return Pearson product-moment correlation coefficient"""
    aligned_signals = _align(reference, sample)
    return np.corrcoef(aligned_signals['reference'], aligned_signals['sample'])[0, 1]


def compute_index_of_agreement(reference, sample):
    """Given reference and sample signals, return Index of Agreement (Willmott 1981)"""
    aligned_signals = _align(reference, sample)

    ref_mean = aligned_signals['reference'].mean()
    num = ((aligned_signals['sample'] - aligned_signals['reference'])**2).sum()
    denom_a = (aligned_signals['sample'] - aligned_signals['reference']).abs()
    denom_b = (aligned_signals['reference'] - ref_mean)**2
    denom = (denom_a + denom_b).sum()
    # handle underfloat
    if denom < 1e-16:
        return 1
    return 1 - num/denom


def compute_mean_square_error(reference, sample, centered=True):
    """Given reference and sample signals, return mean square error (MSE)"""
    aligned_signals = _align(reference, sample)

    error = aligned_signals['reference'] - aligned_signals['sample']
    if centered:
        error += -aligned_signals['reference'].mean() + aligned_signals['sample'].mean()
    return (error**2).mean()


def compute_murphy_skill_score(reference, sample, reference_model=None):
    """Given reference and sample signals, return Murphy Skill Score (Murphy 1988)"""
    # if a reference forecast is not available, use mean of the *original* observations
    if not reference_model:
        reference_model = reference.copy()
        reference_model[:] = reference.mean().values[0]
        reference_model.rename(columns={'reference': 'reference_model'})

    mse_model = compute_mean_square_error(reference, sample, centered=False)
    mse_reference_model = compute_mean_square_error(reference_model, reference, centered=False)
    if mse_reference_model <= 0:
        return -1
    return 1 - mse_model / mse_reference_model


def compute_root_mean_square_error(reference, sample, centered=True):
    """Given reference and sample signals, return Root Mean Square Error (RMSE)"""
    mse = compute_mean_square_error(reference, sample, centered=centered)
    return np.sqrt(mse)


def compute_descriptive_statistics(sample, ddof=0):
    """Given reference and sample signals, return the standard deviation"""
    return (np.max(sample), np.min(sample), np.mean(sample), np.std(sample, ddof=ddof))


def compute_stats(reference, sample):
    """Compute stats and return as DataFrame"""

    return {
        'bias': compute_bias(reference, sample),
        'corr': compute_correlation_coefficient(reference, sample),
        'ioa': compute_index_of_agreement(reference, sample),
        'mse': compute_mean_square_error(reference, sample),
        'mss': compute_murphy_skill_score(reference, sample),
        'rmse': compute_root_mean_square_error(reference, sample),
        'descriptive': compute_descriptive_statistics(sample)
    }
