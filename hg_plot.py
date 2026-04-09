import numpy as np
import matplotlib.pyplot as plt

def hg(g, theta):
    return (1 - g**2) / (4 * np.pi * (1 + g**2 - 2*g*np.cos(theta))**1.5)

theta = np.linspace(0, 2*np.pi, 1000)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

for g, ls in [(-0.5, '--'), (0, '-'), (0.5, '-.'), (0.85, '-')]:
    r = hg(g, theta)
    ax.plot(theta, r, ls, label=f'$g = {g}$', linewidth=2.5)

ax.set_rscale('log')
ax.set_rlim(0.01, 10)
ax.set_rticks([0.01, 0.1, 1, 10])
ax.set_yticklabels(['0.01', '0.1', '1', '10'], fontsize=13)
ax.tick_params(axis='y', pad=15)
ax.tick_params(axis='x', labelsize=14)
ax.set_theta_zero_location('E')
ax.legend(loc='upper left', bbox_to_anchor=(1.08, 1.05), fontsize=14, framealpha=0.9)
plt.tight_layout()
plt.savefig('hg_lobes.pdf', bbox_inches='tight')
plt.savefig('hg_lobes.png', bbox_inches='tight', dpi=300)
