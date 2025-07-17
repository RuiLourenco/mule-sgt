import matplotlib.pyplot as plt

# Data series Sideboard
# x1 = [0.0048,0.0196,0.0910,0.7535]
# y1 = [25.04,28.35,33.42,42.56]
# x2 = [0.0050,0.0175,0.0861,0.6041]
# y2 = [24.7777, 28.2193, 33.8725, 42.5039]


#Data Series Greek
# x3 = [0.0056, 0.0207, 0.0918, 0.5885]
# y3 = [34.0033, 38.5841,43.4458,48.3121]
# x2 = [0.0056, 0.0208, 0.0941, 0.6324]
# y2 = [33.9555, 38.4294, 43.1904, 47.9668]
x1 = [0.0050, 0.0178, 0.0886, 0.7497]
y1 = [34.52, 38.56, 43.01, 48.29]
x2 = [0.0055587, 0.020702, 0.092822, 0.5971]  # Rate in reverse order
y2 = [34.018, 38.531, 43.421, 48.4]
x3 = [0.0055745, 0.020784, 0.09289, 0.5887] # Rate in reverse order
y3 = [34.008, 38.58, 43.47, 48.416] 
x4 = [0.0054521, 0.020036, 0.091224, 0.5862]  # Rate in reverse order
y4 = [34.068, 38.575, 43.385, 47.015]
x5 = [0.0056,0.0209,0.0958,0.6463]
y5 = [33.9099,38.3226,43.0369,46.4706]

# Labels and title
xLabel = 'Rate (bpp)'
yLabel = 'PSNR (dB)'
title = 'RD Comparison - Greek'

# Plotting the data
plt.semilogx(x1, y1, marker='o', label='Mule-Slant', linestyle='--')
plt.semilogx(x2, y2, marker='v', label='Mule-SGT No Refinement')
plt.semilogx(x3, y3, marker='v', label='Mule-SGT Refinement 0.1')
plt.semilogx(x5, y5, marker='x', label='Mule-SGT Sideboard, No Refinement')
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
plt.savefig('rd_curves_greek.png')