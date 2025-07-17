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
x2 = [0.0056623, 0.021416, 0.093736, 0.62349]  # Rate
y2 = [33.813, 38.349, 43.22, 48.389]           # Mean_PSNR_Y


x3 = [0.0055941, 0.020822, 0.091057, 0.6116]  # Rate
y3 = [33.952, 38.451, 43.262, 48.396]         # Mean_PSNR_Y

x4 = [0.0055745, 0.020784, 0.09289, 0.5887]  # Rate
y4 = [34.008, 38.58, 43.47, 48.416]          # Mean_PSNR_Y
x5 = [0.0054581, 0.019919, 0.087017, 0.57528]  # Rate
y5 = [34.138, 38.663, 43.464, 48.417]          # Mean_PSNR_Y
# Hardcoded data ordered from lowest to highest
x6 = [0.005532, 0.020391, 0.093355, 0.62749]  # Rate
y6 = [33.953, 38.432, 43.209, 48.276]  


# Labels and title
xLabel = 'Rate (bpp)'
yLabel = 'PSNR (dB)'
title = 'RD Comparison - Greek'

# Plotting the data
plt.semilogx(x1, y1, marker='o', label='Mule-Slant', linestyle='--')
plt.semilogx(x5, y5, marker='x', label=r'RefinedGridSearch SGT --- $\tau_\sigma=inf$')
plt.semilogx(x3, y3, marker='v', label=r'logdetdiv SGT --- $\tau_\sigma=inf$')
plt.semilogx(x6, y6, marker='v', label=r'Structure Tensor SGT --- $\tau_\sigma=inf$')

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