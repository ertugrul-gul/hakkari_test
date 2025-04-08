import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import io

# --- Veri Girişi ---
data_string = """ID	EC (µS/cm)	Na (meq/l)	Ca (meq/l)	Mg (meq/l)	SAR
DS1	1	333.00	11.947	33.408	12.189	2.502
DS2	2	640.00	16.626	79.789	18.779	2.368
DS4	3	603.00	18.296	81.221	20.502	2.565
DS5	4	1356.00	175.386	67.060	18.738	26.778
DS6	5	841.00	82.363	79.964	17.446	11.802
DS7	6	640.00	45.984	73.920	17.602	6.798
DR1	7	334.00	5.618	49.221	11.567	1.019
DR2	8	347.00	7.663	49.273	12.025	1.384
DR3	9	353.00	8.319	49.788	12.459	1.491
DR4	10	355.00	8.652	47.916	12.302	1.577
DR5	11	379.00	11.112	52.591	13.491	1.933
DR6	12	361.00	8.867	48.424	12.271	1.610
DR7	13	406.00	11.553	52.707	13.241	2.012
DR8	14	924.00	87.723	59.999	12.522	14.568
DR9	15	450.00	17.352	51.711	13.126	3.048
HG1	16	513.00	27.166	47.171	17.179	4.789
HG2	17	1098.00	79.318	62.115	13.993	12.858
HG3	18	795.00	66.405	56.761	15.610	11.039
HG4	19	1023.00	92.585	63.340	13.671	14.920
HG5	20	1509.00	103.898	133.459	41.420	11.111
HG6	21	592.00	22.662	76.561	21.824	3.231
"""

data = pd.read_csv(io.StringIO(data_string), sep='\t')
data.columns = data.columns.str.strip()
ids = data['ID'].tolist()
ec_values = data['EC (µS/cm)'].tolist()
sar_values = data['SAR'].tolist()

# --- Sınıf Sınırı Fonksiyonu ---
def get_sar_boundary(ec, ec_points, sar_points):
    return np.interp(ec, ec_points, sar_points)

# --- USSL Diyagramını Çizme Fonksiyonu ---
def plot_ussl_diagram_detailed(ec_values, sar_values, point_labels):
    ec_boundaries_c = [100, 250, 750, 2250, 5000]
    ec_s_boundary_points = np.array(ec_boundaries_c)
    sar_s1_s2 = np.array([10.0, 10.0, 7.2, 4.8, 2.6])  # C1 içinde S1/S2 sınırı düz (SAR=10) gibi görünüyor
    sar_s2_s3 = np.array([18.0, 18.0, 13.5, 8.6, 4.9])  # C1 içinde S2/S3 sınırı düz (SAR=18) gibi görünüyor
    sar_s3_s4 = np.array([26.0, 26.0, 19.5, 12.4, 7.0])  #

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_xlim(ec_boundaries_c[0], ec_boundaries_c[-1])
    ax.set_ylim(0, 30)
    ax.set_xlabel('Salinity hazard (EC, µS/cm at 25°C)', fontsize=12)
    ax.set_ylabel('Sodium (alkali) hazard (SAR)', fontsize=12)

    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Specific conductance, in µS/cm at 25°C', fontsize=12)

    ax.xaxis.set_major_locator(mticker.FixedLocator(ec_boundaries_c[:-1]))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.xaxis.set_major_locator(mticker.FixedLocator([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format(int(x), ',')))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax3.yaxis.set_minor_locator(mticker.MultipleLocator(2))

    ax.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    for i in range(1, len(ec_boundaries_c) - 1):
        ax.axvline(x=ec_boundaries_c[i], color='black', linestyle='-', linewidth=1.0)

    ax.plot(ec_s_boundary_points, sar_s1_s2, 'k-', linewidth=1.5)
    ax.plot(ec_s_boundary_points, sar_s2_s3, 'k-', linewidth=1.5)
    ax.plot(ec_s_boundary_points, sar_s3_s4, 'k-', linewidth=1.5)

    c_labels_short = ['C1', 'C2', 'C3', 'C4']
    s_labels_short = ['S1', 'S2', 'S3', 'S4']
    y_max_plot = ax.get_ylim()[1]

    for i in range(len(c_labels_short)):
        ec_min = ec_boundaries_c[i]
        ec_max = ec_boundaries_c[i+1]
        ec_center = np.sqrt(ec_min * ec_max)

        sar_upper = get_sar_boundary(ec_center, ec_s_boundary_points, sar_s1_s2)
        ax.text(ec_center, sar_upper / 2, f'{c_labels_short[i]}-{s_labels_short[0]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

        sar_lower = sar_upper
        sar_upper = get_sar_boundary(ec_center, ec_s_boundary_points, sar_s2_s3)
        ax.text(ec_center, (sar_lower + sar_upper) / 2, f'{c_labels_short[i]}-{s_labels_short[1]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

        sar_lower = sar_upper
        sar_upper = get_sar_boundary(ec_center, ec_s_boundary_points, sar_s3_s4)
        ax.text(ec_center, (sar_lower + sar_upper) / 2, f'{c_labels_short[i]}-{s_labels_short[2]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

        sar_lower = sar_upper
        sar_upper = y_max_plot
        ax.text(ec_center, (sar_lower + sar_upper) / 2, f'{c_labels_short[i]}-{s_labels_short[3]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

    # Veri noktalarını çiz ve etiketle
    ax.scatter(ec_values, sar_values, c='red', marker='o', edgecolor='black', s=60, label='Su Örnekleri', zorder=5)
    for i, txt in enumerate(point_labels):
        ax.annotate(txt, (ec_values[i], sar_values[i]), textcoords="offset points", xytext=(0,5),
                    ha='center', va='bottom', fontsize=8, zorder=6)

    plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.9])
    plt.show()

# --- Sınıflandırma Fonksiyonu ---
def classify_sample_detailed(ec, sar):
    ec_boundaries_c = [250, 750, 2250, 5000, 10000]
    ec_s_boundary_points = np.array(ec_boundaries_c)
    sar_s1_s2 = np.array([10.0, 7.2, 4.8, 2.6, 1.5])
    sar_s2_s3 = np.array([18.0, 13.5, 8.6, 4.9, 3.0])
    sar_s3_s4 = np.array([26.0, 19.5, 12.4, 7.0, 4.4])

    if ec < 250:
        c_class = 'C1'
    elif ec < 750:
        c_class = 'C2'
    elif ec < 2250:
        c_class = 'C3'
    else:
        c_class = 'C4'

    sar_b12 = get_sar_boundary(ec, ec_s_boundary_points, sar_s1_s2)
    sar_b23 = get_sar_boundary(ec, ec_s_boundary_points, sar_s2_s3)
    sar_b34 = get_sar_boundary(ec, ec_s_boundary_points, sar_s3_s4)

    if sar < sar_b12:
        s_class = 'S1'
    elif sar < sar_b23:
        s_class = 'S2'
    elif sar < sar_b34:
        s_class = 'S3'
    else:
        s_class = 'S4'

    return f"{c_class}-{s_class}"

# --- Ana Çalıştırma ---
plot_ussl_diagram_detailed(ec_values, sar_values, ids)

print("\nUSSL Diyagramı (Detaylı Görünüm) Oluşturuldu.")
print(f"{len(ec_values)} adet su örneği grafiğe eklendi.")

print("\nÖrneklerin Sınıflandırılması (Detaylı Sınırlara Göre):")
for i in range(len(ids)):
    classification = classify_sample_detailed(ec_values[i], sar_values[i])
    print(f"  {ids[i]}: EC={ec_values[i]:.2f}, SAR={sar_values[i]:.3f} -> Sınıf: {classification}")
