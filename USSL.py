import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import io # Veriyi okumak için

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

# Veriyi pandas DataFrame'e oku
data = pd.read_csv(io.StringIO(data_string), sep='\t')
data.columns = data.columns.str.strip()
ids = data['ID'].tolist()
ec_values = data['EC (µS/cm)'].tolist()
sar_values = data['SAR'].tolist()

# --- Global Değişkenler ve Yardımcı Fonksiyonlar ---
# (Hem çizim hem de sınıflandırma fonksiyonlarının erişebilmesi için dışarı alındı)

# C sınıfı dikey çizgileri için sınırlar (Görsel referansa göre)
ec_boundaries_c = [100, 250, 750, 2250, 5000]

# S Sınıfları için eğimli sınır çizgileri için EC noktaları
ec_s_boundary_points = np.array(ec_boundaries_c)

# Sınır çizgilerinin bu EC noktalarındaki yaklaşık SAR değerleri (Görselden)
# EC=100 için değerler eklendi
sar_s1_s2 = np.array([10.0, 10.0, 7.2, 4.8, 2.6]) # C1 içinde S1/S2 sınırı düz (SAR=10) gibi görünüyor
sar_s2_s3 = np.array([18.0, 18.0, 13.5, 8.6, 4.9])# C1 içinde S2/S3 sınırı düz (SAR=18) gibi görünüyor
sar_s3_s4 = np.array([26.0, 26.0, 19.5, 12.4, 7.0])# C1 içinde S3/S4 sınırı düz (SAR=26) gibi görünüyor

# Interpolasyon fonksiyonu (Sınır çizgilerini herhangi bir EC için hesaplamak için)
def get_sar_boundary(ec, ec_points, sar_points):
    """Verilen EC değeri için sınır çizgisinin SAR değerini enterpolasyonla bulur."""
    # numpy.interp logaritmik x ekseniyle iyi çalışır
    # Ancak sınırlarımızın sol kısmı düz olduğu için, 100-250 aralığını ayrı ele alabiliriz
    if ec < ec_points[1]: # Eğer EC, ilk bölümdeyse (örn. 100-250 aralığı)
        return sar_points[1] # O bölümdeki sabit SAR değerini döndür (ilk iki nokta aynı SAR'da)
    else:
        return np.interp(ec, ec_points[1:], sar_points[1:]) # Diğer kısımlarda enterpolasyon yap

# --- USSL Diyagramını Çizme Fonksiyonu ---
def plot_ussl_diagram_detailed(ec_values, sar_values, point_labels):
    """
    Verilen EC ve SAR değerleri ile detaylı USSL diyagramını çizer (görsel referansa benzer).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Eksen Ayarları ---
    ax.set_xscale('log')
    ax.set_yscale('linear')
    # Eksen sınırlarını global değişkenlerden alalım
    ax.set_xlim(ec_boundaries_c[0], ec_boundaries_c[-1])
    ax.set_ylim(0, 30)

    # Eksen Başlıkları
    ax.set_xlabel('Salinity hazard (EC, µS/cm at 25°C)', fontsize=12)
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('Sodium (alkali) hazard (SAR)', fontsize=12)

    # Üst X ekseni
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Specific conductance, in µS/cm at 25°C', fontsize=12)

    # Tick Ayarları
    # Alt X ekseni ana çizgileri: 100, 250, 750, 2250
    main_ticks_x = ec_boundaries_c[:-1]
    ax.xaxis.set_major_locator(mticker.FixedLocator(main_ticks_x))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # Alt X ekseni ara çizgileri (logaritmik)
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=15))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # Üst X ekseni ana çizgileri
    upper_ticks_x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
    ax2.xaxis.set_major_locator(mticker.FixedLocator(upper_ticks_x))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}")) # Binlik ayraç
    ax2.xaxis.set_tick_params(which='major', pad=7) # Üstteki etiketler için boşluk

    # Sol Y ekseni çizgileri
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")

    # Sağ Y ekseni çizgileri
    ax3 = ax.twinx()
    ax3.set_yscale('linear')
    ax3.set_ylim(ax.get_ylim())
    ax3.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax3.yaxis.set_minor_locator(mticker.MultipleLocator(2))

    # İç gridleri kapat
    ax.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    # --- C Sınır Çizgilerini Çizme (Dikey) ---
    # 100, 250, 750, 2250 sınırlarını çiz (ilk ve son hariç)
    for i in range(1, len(ec_boundaries_c) - 1):
         ax.axvline(x=ec_boundaries_c[i], color='black', linestyle='-', linewidth=1.0)

    # --- S Sınır Çizgilerini Çizme (Eğimli/Düz) ---
    # Daha düzgün çizim için daha fazla nokta ile interpolasyon yapabiliriz
    ec_plot_points = np.logspace(np.log10(ec_boundaries_c[0]), np.log10(ec_boundaries_c[-1]), 100)
    sar_plot_s12 = [get_sar_boundary(ec, ec_s_boundary_points, sar_s1_s2) for ec in ec_plot_points]
    sar_plot_s23 = [get_sar_boundary(ec, ec_s_boundary_points, sar_s2_s3) for ec in ec_plot_points]
    sar_plot_s34 = [get_sar_boundary(ec, ec_s_boundary_points, sar_s3_s4) for ec in ec_plot_points]

    ax.plot(ec_plot_points, sar_plot_s12, 'k-', linewidth=1.5)
    ax.plot(ec_plot_points, sar_plot_s23, 'k-', linewidth=1.5)
    ax.plot(ec_plot_points, sar_plot_s34, 'k-', linewidth=1.5)

    # --- Bölge Etiketlerini Ekleme (C1-S1 vb.) ---
    c_labels_short = ['C1', 'C2', 'C3', 'C4']
    s_labels_short = ['S1', 'S2', 'S3', 'S4']
    y_max_plot = ax.get_ylim()[1]

    for i in range(len(c_labels_short)): # C1'den C4'e
        ec_min = ec_boundaries_c[i]
        ec_max = ec_boundaries_c[i+1]
        ec_center = np.sqrt(ec_min * ec_max) # Logaritmik orta nokta

        # S1 bölgesi
        sar_lower = 0
        sar_upper = get_sar_boundary(ec_center, ec_s_boundary_points, sar_s1_s2)
        sar_center = (sar_lower + sar_upper) / 2
        if sar_center < 1 : sar_center=1 # Çok aşağıya gitmesin
        ax.text(ec_center, sar_center, f'{c_labels_short[i]}-{s_labels_short[0]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

        # S2 bölgesi
        sar_lower = sar_upper
        sar_upper = get_sar_boundary(ec_center, ec_s_boundary_points, sar_s2_s3)
        sar_center = (sar_lower + sar_upper) / 2
        ax.text(ec_center, sar_center, f'{c_labels_short[i]}-{s_labels_short[1]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

        # S3 bölgesi
        sar_lower = sar_upper
        sar_upper = get_sar_boundary(ec_center, ec_s_boundary_points, sar_s3_s4)
        sar_center = (sar_lower + sar_upper) / 2
        ax.text(ec_center, sar_center, f'{c_labels_short[i]}-{s_labels_short[2]}',
                ha='center', va='center', fontsize=11, fontweight='bold')

        # S4 bölgesi
        sar_lower = sar_upper
        sar_upper = y_max_plot
        sar_center = (sar_lower + sar_upper) / 2
        if sar_center < sar_lower + 1: sar_center = sar_lower + 1
        if sar_center > y_max_plot -1: sar_center = y_max_plot - 1
        ax.text(ec_center, sar_center, f'{c_labels_short[i]}-{s_labels_short[3]}',
                ha='center', va='center', fontsize=11, fontweight='bold')


    # --- Sınıf Etiketlerini Ekleme (Low, Medium, High, Very High) ---
    # (Yorum satırı kaldırıldı)
    # Alt Kısım (Salinity Hazard - C Sınıfları)
    c_class_desc = ['Low', 'Medium', 'High', 'Very High']
    y_pos_c_label = -3 # Y ekseninde mutlak konum (0'ın biraz altı)
    for i in range(len(c_labels_short)):
        ec_min = ec_boundaries_c[i]
        ec_max = ec_boundaries_c[i+1]
        ec_center = np.sqrt(ec_min * ec_max)
        ax.text(ec_center, y_pos_c_label, f'{c_labels_short[i]}\n{c_class_desc[i]}',
                ha='center', va='top', fontsize=9, # Biraz küçülttük
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='black'))

    # Sol Kısım (Sodium Hazard - S Sınıfları)
    s_class_desc = ['Low', 'Medium', 'High', 'Very High']
    s_mid_points = [5, 14, 22, 28] # Sınıfların yaklaşık orta SAR değerleri
    # x_pos_s_label = 80 # Bu log skalada çok sola düşebilir, göreceli konum deneyelim
    # x ekseninin sol kenar boşluğuna göre ayarlama yapalım
    # fig.subplotpars.left kullanarak sol boşluğu alıp ona göre yerleştirelim (daha karmaşık)
    # Şimdilik sabit bir EC değeri deneyelim (örn. 85)
    x_pos_s_label = 85 # Eksenin soluna (log skalada 100'ün biraz solu)
    for i in range(len(s_labels_short)):
        # Etiketlerin sadece grafik alanı içinde kalmasını sağlayalım (opsiyonel)
        # if s_mid_points[i] > ax.get_ylim()[0] and s_mid_points[i] < ax.get_ylim()[1]:
         ax.text(x_pos_s_label, s_mid_points[i], f'{s_class_desc[i]}\n{s_labels_short[i]}',
                 ha='center', va='center', fontsize=9, # Biraz küçülttük
                 bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='black'))


    # --- Veri Noktalarını Çizme ---
    # (Fonksiyon içine taşındı)
    ax.scatter(ec_values, sar_values, c='red', marker='o', edgecolor='black', s=60, label='Su Örnekleri', zorder=5)

    # --- Veri Noktalarını Etiketleme ---
    # (Fonksiyon içine taşındı)
    for i, txt in enumerate(point_labels):
        ax.annotate(str(txt), (ec_values[i], sar_values[i]), textcoords="offset points", xytext=(0,5),
                   ha='center', va='bottom', fontsize=8, zorder=6)

    # --- Layout ve Gösterim ---
    # (Fonksiyon içine taşındı)
    plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.92]) # Etiketlere yer açmak için kenar boşlukları ayarlandı
    plt.show()


# --- Sınıflandırma Fonksiyonu (Global değişkenleri kullanır) ---
def classify_sample_detailed(ec, sar):
    """Bir su örneğini USSL sınıfına göre sınıflandırır."""
    # C Sınıfı
    if ec < ec_boundaries_c[1]: c_class = 'C1'     # < 250
    elif ec < ec_boundaries_c[2]: c_class = 'C2' # < 750
    elif ec < ec_boundaries_c[3]: c_class = 'C3' # < 2250
    else: c_class = 'C4'                         # >= 2250

    # S Sınıfı (Global fonksiyon ve değişkenleri kullanır)
    sar_b12 = get_sar_boundary(ec, ec_s_boundary_points, sar_s1_s2)
    sar_b23 = get_sar_boundary(ec, ec_s_boundary_points, sar_s2_s3)
    sar_b34 = get_sar_boundary(ec, ec_s_boundary_points, sar_s3_s4)

    if sar < sar_b12: s_class = 'S1'
    elif sar < sar_b23: s_class = 'S2'
    elif sar < sar_b34: s_class = 'S3'
    else: s_class = 'S4'

    return f"{c_class}-{s_class}"

# --- Ana Kodu Çalıştırma ---
# (Doğru girintileme seviyesinde)
plot_ussl_diagram_detailed(ec_values, sar_values, ids)

print("\nUSSL Diyagramı (Detaylı Görünüm) Oluşturuldu.")
print(f"{len(ec_values)} adet su örneği grafiğe eklendi.")

print("\nÖrneklerin Sınıflandırılması (Detaylı Sınırlara Göre):")
for i in range(len(ids)):
    # ID'yi de alalım
    sample_id = ids[i]
    sample_ec = ec_values[i]
    sample_sar = sar_values[i]
    classification = classify_sample_detailed(sample_ec, sample_sar)
    print(f"  {sample_id}: EC={sample_ec:.2f}, SAR={sample_sar:.3f} -> Sınıf: {classification}")