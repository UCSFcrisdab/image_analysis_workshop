"""Build a sawtooth wave from its first 10 sine waves, and plot the recipe.

Run from the top folder of the repository with:

    uv run src/sawtooth_demo.py

The figure is shown in a window and saved as
Session4_FourierAndNoise/Cristina_Session4/sawtooth_demo.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

out_path = Path("Session4_FourierAndNoise/Cristina_Session4/sawtooth_demo.png")

N_WAVES = 10
PERIOD = 1.0  # seconds
time = np.linspace(0, 2 * PERIOD, 5000)  # show two repeats

# The sawtooth: rises steadily from -1 to +1 during each period, then drops back at once
sawtooth = 2 * ((time / PERIOD) % 1) - 1

# The recipe.  A sawtooth contains ALL multiples of its basic frequency (1, 2, 3, ... Hz,
# not just the odd ones like a square wave), with amplitude 2 / (pi x multiple).
# A rising sawtooth is minus a sum of sines, and "minus" means a phase of 180 degrees.
multiples = np.arange(1, N_WAVES + 1)
freqs = multiples / PERIOD
amps = 2 / (np.pi * multiples)
phases = np.full(N_WAVES, np.pi)


def sine(t, freq, amp, phase):
    return amp * np.sin(2 * np.pi * freq * t + phase)


waves = [sine(time, f, a, p) for f, a, p in zip(freqs, amps, phases)]
total = np.sum(waves, axis=0)

print(f"{'frequency':>10} {'amplitude':>10} {'phase':>8}")
for f, a, p in zip(freqs, amps, phases):
    print(f"{f:>7g} Hz {a:>10.3f} {np.degrees(p):>7.0f}°")

colors = plt.cm.tab10(np.arange(N_WAVES))

fig = plt.figure(figsize=(12, 6.5))
grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.45, wspace=0.25)
ax_sum = fig.add_subplot(grid[0, 0])
ax_parts = fig.add_subplot(grid[1, 0])
ax_amp = fig.add_subplot(grid[0, 1])
ax_phase = fig.add_subplot(grid[1, 1])

for ax, title in [(ax_sum, f"Sum of the first {N_WAVES} sine waves"),
                  (ax_parts, "The 10 waves that are added")]:
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax_parts.set_xlabel("time (s)")

ax_sum.plot(time, sawtooth, color="#555555", lw=1.5, ls="--", label="sawtooth wave")
ax_sum.plot(time, total, color="#2a78d6", lw=2, label="sum")
ax_sum.legend(loc="upper right", frameon=False, fontsize=9)
for wave, color, f in zip(waves, colors, freqs):
    ax_parts.plot(time, wave, color=color, lw=1.3, label=f"{f:g} Hz")

# Stem plots: one stem per wave, at its frequency
for ax in (ax_amp, ax_phase):
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlim(0, N_WAVES + 1)
    ax.set_xticks(range(0, N_WAVES + 1, 2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for f, a, p, c in zip(freqs, amps, phases, colors):
    ax_amp.plot([f, f], [0, a], color=c, lw=2)
    ax_amp.plot(f, a, "o", color=c, ms=7)
    ax_phase.plot([f, f], [0, np.degrees(p)], color=c, lw=2)
    ax_phase.plot(f, np.degrees(p), "o", color=c, ms=7)
ax_amp.set_ylim(0, 0.8)
ax_amp.set_title("Amplitude of each wave", loc="left", fontsize=11)
ax_phase.set_ylim(-200, 200)
ax_phase.set_yticks([-180, -90, 0, 90, 180])
ax_phase.set_ylabel("degrees")
ax_phase.set_title("Phase of each wave", loc="left", fontsize=11)
ax_phase.set_xlabel("frequency (Hz)")

fig.suptitle("A sawtooth wave = a sum of sine waves, each with its own "
             "frequency, amplitude and phase", fontsize=12)

out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=100)
print(f"Saved {out_path}")
plt.show()
