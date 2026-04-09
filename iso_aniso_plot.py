import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rcParams

# ---------------------------
# Styling (thesis-quality) - RESTORED
# ---------------------------
rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.linewidth": 1.2,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ---------------------------
# Mahalanobis distance
# ---------------------------
def mahalanobis_2d(x, y, mean, cov):
    pos = np.dstack((x, y))
    diff = pos - mean
    inv_cov = np.linalg.inv(cov)
    return np.einsum('...i,ij,...j->...', diff, inv_cov, diff)

# ---------------------------
# Draw covariance ellipse + axes
# ---------------------------
def draw_cov_ellipse(ax, mean, cov, chi2_val=2.30, color='red'):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigvals * chi2_val)

    ellipse = Ellipse(mean, width, height,
                      angle=angle,
                      fill=False,
                      edgecolor=color,
                      linewidth=2.5)
    ax.add_patch(ellipse)

    # principal axes
    for i in range(2):
        vec = eigvecs[:, i]
        length = np.sqrt(eigvals[i] * chi2_val)
        ax.plot([mean[0], mean[0] + vec[0]*length],
                [mean[1], mean[1] + vec[1]*length],
                color=color, # Kept original red
                linewidth=2.5)

# ---------------------------
# Compute tight plot limits
# ---------------------------
def get_extent(cov, chi2_val):
    eigvals = np.linalg.eigvalsh(cov)
    # Use the max eigenvalue and the max contour level (6.0) to find the boundary
    return np.sqrt(eigvals.max() * chi2_val)

# ---------------------------
# Setup Gaussians
# ---------------------------
mean = np.array([0, 0])

cov_iso = np.eye(2)
cov_aniso = np.array([[2.0, 0.8],
                      [0.8, 0.5]])

# FIX: Calculate extent based on the maximum contour level (6.0) 
# instead of the ellipse level (2.30) to prevent clipping.
max_contour_level = 6.0
extent = max(get_extent(cov_iso, max_contour_level), 
             get_extent(cov_aniso, max_contour_level))

padding = 1.05 # Minimal padding to keep Gaussians large
limit = extent * padding

# Grid (tight around ellipse)
x = np.linspace(-limit, limit, 400)
y = np.linspace(-limit, limit, 400)
X, Y = np.meshgrid(x, y)

# Fields
Z_iso = mahalanobis_2d(X, Y, mean, cov_iso)
Z_aniso = mahalanobis_2d(X, Y, mean, cov_aniso)

levels = np.linspace(0, 6, 25)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax in axes:
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

# ---- Isotropic
cf1 = axes[0].contourf(X, Y, Z_iso, levels=levels, cmap="viridis")
axes[0].contour(X, Y, Z_iso, levels=levels, colors='black', linewidths=0.5)
axes[0].set_title(r"Isotropic Gaussian ($\Sigma = \sigma^2 I$)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
draw_cov_ellipse(axes[0], mean, cov_iso)

# ---- Anisotropic
cf2 = axes[1].contourf(X, Y, Z_aniso, levels=levels, cmap="viridis")
axes[1].contour(X, Y, Z_aniso, levels=levels, colors='black', linewidths=0.5)
axes[1].set_title(r"Anisotropic Gaussian ($\Sigma \neq \sigma^2 I$)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
draw_cov_ellipse(axes[1], mean, cov_aniso)

# Colorbar
cbar = fig.colorbar(cf2, ax=axes, shrink=0.9)
cbar.set_label("Mahalanobis distance")

# ---------------------------
# Save (PDF for thesis)
# ---------------------------
plt.savefig("gaussians_final.pdf",
            bbox_inches='tight',
            transparent=True)

plt.show()