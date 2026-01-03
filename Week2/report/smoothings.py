import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def exponential_moving_average(values, alpha=0.9):
    """
    Exponential moving average (EMA) smoothing.
    
    Args:
        values: Array of values to smooth
        alpha: Smoothing factor (0 < alpha < 1). Higher = less smoothing.
               Common values: 0.9 (light), 0.8 (moderate), 0.6 (heavy)
    
    Returns:
        Smoothed array
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def gaussian_smooth(values, sigma=2):
    """
    Gaussian smoothing using a Gaussian kernel.
    
    Args:
        values: Array of values to smooth
        sigma: Standard deviation of Gaussian kernel. Higher = more smoothing.
               Common values: 1 (light), 2 (moderate), 3-5 (heavy)
    
    Returns:
        Smoothed array
    """
    return gaussian_filter1d(values, sigma=sigma, mode='nearest')


def moving_average(values, window_size=5):
    """
    Simple moving average (uniform kernel).
    
    Args:
        values: Array of values to smooth
        window_size: Size of the averaging window (must be odd for symmetry)
    
    Returns:
        Smoothed array
    """
    if window_size % 2 == 0:
        window_size += 1  # Make odd for symmetry
    
    kernel = np.ones(window_size) / window_size
    # Use 'same' mode to keep same length, pad edges
    smoothed = np.convolve(values, kernel, mode='same')
    
    # Fix edge effects
    for i in range(window_size // 2):
        smoothed[i] = np.mean(values[:i + window_size // 2 + 1])
        smoothed[-(i+1)] = np.mean(values[-(i + window_size // 2 + 1):])
    
    return smoothed


def savitzky_golay_smooth(values, window_size=11, poly_order=3):
    """
    Savitzky-Golay filter - fits successive windows with a polynomial.
    Good at preserving peaks while smoothing.
    
    Args:
        values: Array of values to smooth
        window_size: Size of the filtering window (must be odd)
        poly_order: Order of polynomial to fit (must be < window_size)
    
    Returns:
        Smoothed array
    """
    if window_size % 2 == 0:
        window_size += 1
    
    if poly_order >= window_size:
        poly_order = window_size - 1
    
    return savgol_filter(values, window_size, poly_order, mode='nearest')


def median_filter_smooth(values, window_size=5):
    """
    Median filter - replaces each value with the median of its window.
    Very robust to outliers/spikes.
    
    Args:
        values: Array of values to smooth
        window_size: Size of the filtering window
    
    Returns:
        Smoothed array
    """
    from scipy.ndimage import median_filter
    return median_filter(values, size=window_size, mode='nearest')


def lowess_smooth(values, frac=0.1):
    """
    LOWESS (Locally Weighted Scatterplot Smoothing).
    Non-parametric regression - fits local polynomials.
    
    Args:
        values: Array of values to smooth
        frac: Fraction of data to use for each local fit (0 < frac <= 1)
              Common values: 0.05 (light), 0.1 (moderate), 0.2 (heavy)
    
    Returns:
        Smoothed array
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
        x = np.arange(len(values))
        smoothed = sm_lowess(values, x, frac=frac)
        return smoothed[:, 1]  # Return only y values
    except ImportError:
        print("Warning: statsmodels not installed. Install with: pip install statsmodels")
        return gaussian_smooth(values)  # Fallback


def plot_comparison(values, title="Smoothing Comparison", figsize=(15, 10)):
    """
    Plot original values with multiple smoothing methods for comparison.
    
    Args:
        values: Array of values to smooth
        title: Plot title
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    x = np.arange(len(values))
    
    # Original
    axes[0, 0].plot(x, values, alpha=0.3, label='Original')
    axes[0, 0].plot(x, values, 'o', markersize=2, alpha=0.5)
    axes[0, 0].set_title('Original (no smoothing)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # EMA variations
    for idx, alpha in enumerate([0.9, 0.8, 0.6]):
        ax = axes[0, 1] if idx == 0 else (axes[0, 2] if idx == 1 else axes[1, 0])
        smoothed = exponential_moving_average(values, alpha=alpha)
        ax.plot(x, values, alpha=0.2, color='gray')
        ax.plot(x, smoothed, linewidth=2, label=f'Smoothed (α={alpha})')
        ax.set_title(f'Exponential MA (α={alpha})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Gaussian
    ax = axes[1, 1]
    smoothed = gaussian_smooth(values, sigma=2)
    ax.plot(x, values, alpha=0.2, color='gray')
    ax.plot(x, smoothed, linewidth=2, label='Smoothed (σ=2)')
    ax.set_title('Gaussian Smooth (σ=2)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Moving average
    ax = axes[1, 2]
    smoothed = moving_average(values, window_size=7)
    ax.plot(x, values, alpha=0.2, color='gray')
    ax.plot(x, smoothed, linewidth=2, label='Smoothed (win=7)')
    ax.set_title('Moving Average (window=7)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Savitzky-Golay
    ax = axes[2, 0]
    smoothed = savitzky_golay_smooth(values, window_size=11, poly_order=3)
    ax.plot(x, values, alpha=0.2, color='gray')
    ax.plot(x, smoothed, linewidth=2, label='Smoothed')
    ax.set_title('Savitzky-Golay (win=11, poly=3)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Median filter
    ax = axes[2, 1]
    smoothed = median_filter_smooth(values, window_size=5)
    ax.plot(x, values, alpha=0.2, color='gray')
    ax.plot(x, smoothed, linewidth=2, label='Smoothed (win=5)')
    ax.set_title('Median Filter (window=5)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # LOWESS
    ax = axes[2, 2]
    smoothed = lowess_smooth(values, frac=0.1)
    ax.plot(x, values, alpha=0.2, color='gray')
    ax.plot(x, smoothed, linewidth=2, label='Smoothed (frac=0.1)')
    ax.set_title('LOWESS (frac=0.1)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    
    # Generate noisy test data (simulating test loss/accuracy)
    np.random.seed(42)
    n_epochs = 100
    
    # Simulate test loss with trend + noise
    trend = 2.0 * np.exp(-np.arange(n_epochs) / 20) + 0.5
    noise = np.random.normal(0, 0.15, n_epochs)
    test_loss = trend + noise
    
    # Plot comparison
    fig = plot_comparison(test_loss, title="Test Loss Smoothing Methods")
    plt.savefig('smoothing_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved!")
    
    # Show individual example
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(test_loss))
    
    ax.plot(x, test_loss, 'o-', alpha=0.3, markersize=3, label='Raw test loss', color='lightblue')
    ax.plot(x, exponential_moving_average(test_loss, alpha=0.8), 
            linewidth=2.5, label='EMA (α=0.8)', color='darkblue')
    ax.plot(x, gaussian_smooth(test_loss, sigma=2), 
            linewidth=2, label='Gaussian (σ=2)', color='red', linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Smoothed Test Loss Visualization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('example_usage.png', dpi=150, bbox_inches='tight')
    print("Example usage plot saved!")
    
    plt.show()