import matplotlib.pyplot as plt
import numpy as np

# EC ve SAR sınır değerleri
ec_values = [0.1, 0.25, 0.75, 2.25, 5]  # Logaritmik ölçek
sar_values = [0, 10, 18, 26]            # Lineer ölçek

# Sınır çizgileri
plt.plot([0, 40], [ec_values[0], ec_values[0]], 'k--', label='C1')
plt.plot([0, 40], [ec_values[1], ec_values[1]], 'k--', label='C2')
plt.plot([0, 40], [ec_values[2], ec_values[2]], 'k--', label='C3')
plt.plot([0, 40], [ec_values[3], ec_values[3]], 'k--', label='C4')

plt.plot([sar_values[1], sar_values[1]], [0.1, 5], 'r--', label='S1')
plt.plot([sar_values[2], sar_values[2]], [0.1, 5], 'r--', label='S2')
plt.plot([sar_values[3], sar_values[3]], [0.1, 5], 'r--', label='S3')

# Noktalar (örnek veri)
ec_data = [0.5, 1.5, 2.5]
sar_data = [5, 15, 25]

plt.scatter(sar_data, ec_data, color='blue', label='Su Örnekleri')

# Ayarlar
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('SAR')
plt.ylabel('EC (dS/m)')
plt.title('USSL Diyagramı')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.show()
