import matplotlib.pyplot as plt

# Data series Sideboard
x1 = [0.0048,0.0196,0.0910,0.7535]
y1 = [25.04,28.35,33.42,42.56]
x2 = [0.0049137,0.017446,0.085809,0.57946]
y2 = [24.783, 28.303, 34.154, 42.878]
x3 = [0.0052471,0.017543,0.087795,0.57801]
y3 = [24.861, 28.322, 34.277, 43]
x4 = [0.0048214, 0.017779, 0.082651, 0.56118]  # Rate in reverse order
y4 = [24.875, 28.685, 34.341, 40.836]

#Data Series Greek
# x3 = [0.0056, 0.0207, 0.0918, 0.5885]
# y3 = [34.0033, 38.5841,43.4458,48.3121]
# x2 = [0.0056, 0.0208, 0.0941, 0.6324]
# y2 = [33.9555, 38.4294, 43.1904, 47.9668]
# x1 = [0.0050, 0.0178, 0.0886, 0.7497]
# y1 = [34.52, 38.56, 43.01, 48.29]

# Labels and title
xLabel = 'Rate (bpp)'
yLabel = 'PSNR (dB)'
title = 'RD Comparison - Sideboard'

# Plotting the data
plt.semilogx(x1, y1, marker='o', label='Mule-Slant', linestyle='--')
plt.semilogx(x2, y2, marker='v', label='Mule-SGT No Refinement')
plt.semilogx(x3, y3, marker='x', label='Mule-SGT Refinement 0.1')
plt.semilogx(x4, y4, marker='x', label='Mule-SGT Sideboard, Refinement 10-1')
# plt.plot(x1, y1, marker='o', label='Mule-Slant')
# plt.plot(x2, y2, marker='v', label='Mule-SGT No Refinement')
#plt.plot(x2, y2, marker='x', label='Mule-SGT Refinement 0.1')
# plt.plot(x1, y1, marker='o', label='Mule-Slant')
# plt.plot(x2, y2, marker='x', label='Mule-SGT')
# Adding labels and title
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.title(title)

# Adding legend
plt.legend()

# Display the graph
plt.savefig('rd_curves_sideboard.png')