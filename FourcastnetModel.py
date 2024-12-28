import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Veri setini yüklemek ve normalize etmek için özel bir veri seti sınıfı oluşturma
class WeatherDataset(Dataset):
    def __init__(self, file_path, variable_name):
        self.data = xr.open_dataset(file_path)[variable_name].values
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std  # Normalize etme

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def denormalize(self, data):
        return (data * self.std) + self.mean  # Denormalize işlemi

# Basit bir FourCastNet model mimarisi
class FourCastNet(nn.Module):
    def __init__(self):
        super(FourCastNet, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.output(x)
        return x

# Veri Yükleme
file_path_0 = './data_H/data_0.nc'  # Ensure this path is correct and accessible
file_path_1 = './data_H/data_1.nc'  # Ensure this path is correct and accessible
data_0 = xr.open_dataset(file_path_0)
data_1 = xr.open_dataset(file_path_1)
data = xr.concat([data_0, data_1], dim='time')
t2m = data['t2m']  # 2 metre sıcaklık verisi

# Veri setini yükleme
variable_name = 't2m'
dataset = WeatherDataset(file_path_0, variable_name)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model, kayıp fonksiyonu ve optimizer tanımlama
model = FourCastNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim ve doğrulama kayıplarını kaydetmek için listeler
epochs = 10
train_losses = []
val_losses = []

# Modeli eğitme
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs in train_loader:
        inputs = inputs.unsqueeze(1)  # Kanalları ekleme (N, 1, H, W)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Doğrulama
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Eğitim ve doğrulama kayıplarını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

# Significant divergence kontrolü
if abs(train_losses[-1] - val_losses[-1]) > 0.1 * train_losses[-1]:
    plt.annotate("Significant divergence detected", 
                 xy=(epochs, val_losses[-1]), 
                 xytext=(epochs-2, val_losses[-1]+0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=10, color='red')

plt.show()

# Örnek bir tahmini görselleştirme
sample_input = next(iter(val_loader)).unsqueeze(1)[:1]  # Bir örnek seç
model.eval()
with torch.no_grad():
    predicted_output = model(sample_input).squeeze(0).squeeze(0).numpy()
    actual_output = sample_input.squeeze(0).squeeze(0).numpy()

# Denormalize tahmin ve gerçek değerler
denormalized_predicted = dataset.denormalize(predicted_output)
denormalized_actual = dataset.denormalize(actual_output)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(denormalized_actual, cmap="coolwarm")
plt.title("Actual (Denormalized)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(denormalized_predicted, cmap="coolwarm")
plt.title("Predicted (Denormalized)")
plt.colorbar()

plt.show()

# Modeli kaydetme
torch.save(model.state_dict(), "fourcastnet_model.pth")

# 2040 yılı için aylık ortalama sıcaklıkları çizdirme
def plot_monthly_average_temperature(predictions, dataset, year=2040):
    monthly_avg = []
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Gün sayıları
    start_idx = 0

    for days in days_in_month:
        end_idx = start_idx + days
        monthly_avg.append(dataset.denormalize(predictions[start_idx:end_idx]).mean())
        start_idx = end_idx

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 13), monthly_avg, color='skyblue')
    plt.xlabel('Aylar')
    plt.ylabel('Ortalama Sıcaklık (°C)')
    plt.title(f'{year} Yılı Aylık Ortalama Sıcaklıkları')
    plt.xticks(range(1, 13), ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık'])
    plt.show()

# Tahminlerden aylık ortalama sıcaklıkları çizdir
model.eval()
all_predictions = []
with torch.no_grad():
    for inputs in DataLoader(dataset, batch_size=32):
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        all_predictions.append(outputs.squeeze().numpy())

predictions = np.concatenate(all_predictions)
plot_monthly_average_temperature(predictions, dataset)
