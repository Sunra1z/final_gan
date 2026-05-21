import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import hashlib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Создаем папку для сохранения графиков диссертации
os.makedirs("thesis_plots", exist_ok=True)
os.makedirs("./export", exist_ok=True)

# ГИПЕРПАРАМЕТРЫ
MSG_LEN = 24
KEY_LEN = 24
SENSOR_DIM = 561
ALPHABET_K = 1114112
BATCH_SIZE = 128
NUM_EPOCHS = 50  # Установим оптимальное число эпох


# АРХИТЕКТУРА (Алиса, Боб, Ева)
class CryptoGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CryptoGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class CryptoDiscriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CryptoDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Инициализация
alice = CryptoGenerator(MSG_LEN + KEY_LEN + SENSOR_DIM, MSG_LEN)
bob = CryptoGenerator(MSG_LEN + KEY_LEN + SENSOR_DIM, MSG_LEN)
eve = CryptoDiscriminator(MSG_LEN, MSG_LEN)

optimizer_alice_bob = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=0.0008)
optimizer_eve = optim.Adam(eve.parameters(), lr=0.0008)
mse_loss = nn.MSELoss()


# ЗАГРУЗКА ДАННЫХ
def load_sensor_dataset(filepath="X_train.txt"):
    if os.path.exists(filepath):
        print(f"[INFO] Загрузка реального датасета из {filepath}...")
        data = np.loadtxt(filepath)
    else:
        print("[WARNING] Файл датасета не найден. Генерируется Mock-датасет...")
        data = np.random.uniform(-1, 1, (7352, SENSOR_DIM))  # UCI нормализован [-1, 1]

    dataset = TensorDataset(torch.FloatTensor(data))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# КРИПТОГРАФИЯ
def residual_entropy_injection(ai_bits):
    jitter = int(time.time_ns()) % 2
    thermal_noise = np.random.randint(0, 2, len(ai_bits)) ^ jitter
    return np.bitwise_xor(ai_bits, thermal_noise)


def encrypt_text(message, key_tensor, sensor_features):
    alice.eval()
    cipher_tokens, raw_bits_history = [], []
    for char in message:
        char_val = ord(char)
        p_bits = torch.tensor([int(b) for b in bin(char_val)[2:].zfill(MSG_LEN)]).float()
        with torch.no_grad():
            input_data = torch.cat([p_bits.unsqueeze(0), key_tensor, sensor_features], dim=1)
            g_bits = (alice(input_data).numpy().flatten() > 0.5).astype(int)
            secure_bits = residual_entropy_injection(g_bits)
            raw_bits_history.extend(secure_bits)
            g_hash = hashlib.sha256(secure_bits.tobytes()).hexdigest()
            g_val = int(g_hash, 16) % ALPHABET_K
            cipher_tokens.append((char_val + g_val) % ALPHABET_K)
    return cipher_tokens, raw_bits_history


# ОБУЧЕНИЕ И СБОР МЕТРИК
def train_model(dataloader):
    print(f"--- Начало обучения K-GAN (Эпох: {NUM_EPOCHS}) ---")
    alice.train();
    bob.train();
    eve.train()

    # Списки для хранения истории (Визуализация п. 3 и 5)
    history_loss_bob = []
    history_loss_eve = []

    progression_samples = {}

    for epoch in range(NUM_EPOCHS + 1):
        epoch_loss_b, epoch_loss_e = 0.0, 0.0

        # Переменная для сохранения сэмпла шифротекста этой эпохи
        last_cipher_sample = None

        for batch_idx, (sensor_data,) in enumerate(dataloader):
            P = torch.randint(0, 2, (BATCH_SIZE, MSG_LEN)).float()
            K = torch.randint(0, 2, (BATCH_SIZE, KEY_LEN)).float()
            S = sensor_data

            # 1. Обучение Евы
            optimizer_eve.zero_grad()
            C = alice(torch.cat([P, K, S], dim=1))
            P_eve = eve(C.detach())
            loss_e = mse_loss(P_eve, P)
            loss_e.backward();
            optimizer_eve.step()

            # 2. Обучение Алисы и Боба
            optimizer_alice_bob.zero_grad()
            C_shared = alice(torch.cat([P, K, S], dim=1))
            P_bob = bob(torch.cat([C_shared, K, S], dim=1))
            loss_b = mse_loss(P_bob, P)
            loss_e_adv = mse_loss(eve(C_shared), P)
            loss_ab = loss_b + (1.0 - loss_e_adv).pow(2)
            loss_ab.backward();
            optimizer_alice_bob.step()

            epoch_loss_b += loss_b.item()
            epoch_loss_e += loss_e.item()

            if batch_idx == 0:
                last_cipher_sample = C_shared[0].detach().numpy()  # Сохраняем первый шифротекст батча

        avg_loss_b = epoch_loss_b / len(dataloader)
        avg_loss_e = epoch_loss_e / len(dataloader)
        history_loss_bob.append(avg_loss_b)
        history_loss_eve.append(avg_loss_e)


        if epoch in [1, NUM_EPOCHS // 2, NUM_EPOCHS]:
            progression_samples[epoch] = last_cipher_sample

        if epoch % 10 == 0:
            print(f"Эпоха {epoch:3} | Avg Loss Bob: {avg_loss_b:.6f} | Avg Loss Eve: {avg_loss_e:.6f}")


    alice.eval()
    dummy_P = torch.randint(0, 2, (1000, MSG_LEN)).float()
    dummy_K = torch.randint(0, 2, (1000, KEY_LEN)).float()


    dummy_S = dataloader.dataset.tensors[0][:1000]

    generated_cipher = alice(torch.cat([dummy_P, dummy_K, dummy_S], dim=1)).detach().numpy().flatten()

    return history_loss_bob, history_loss_eve, progression_samples, dummy_S.numpy().flatten(), generated_cipher



def visualize_all_results(bob_log, eve_log, prog_samples, real_raw, fake_raw):
    print("\n--- Запуск модуля визуализации результатов ---")

    # Loss-функции и графики обучения
    plt.figure(figsize=(10, 6))
    plt.plot(bob_log, label='Loss Боба (Дешифратор)', color='#1f77b4', linewidth=2.5)
    plt.plot(eve_log, label='Loss Евы (Криптоаналитик)', color='#d62728', linewidth=2.5)
    plt.axhline(y=0.25, color='gray', linestyle='--', label='Слепое угадывание (0.25)')
    plt.title('Прогрессия состязательного обучения K-GAN')
    plt.xlabel('Эпохи');
    plt.ylabel('MSE Loss');
    plt.legend();
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("thesis_plots/1_training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Сравнение Real/Fake данных (Анализ энтропии)
    plt.figure(figsize=(10, 6))
    # UCI HAR данные нормализованы [-1, 1], выход нейросети [0, 1].
    # Для сравнения приведем нейросеть к [-1, 1]
    fake_normalized = (fake_raw * 2) - 1

    sns.kdeplot(real_raw[:2000], fill=True, label="Real (Физические сенсоры UCI HAR)", color="#9467bd", alpha=0.5)
    sns.kdeplot(fake_normalized[:2000], fill=True, label="Generated (Шифротекст K-GAN)", color="#2ca02c", alpha=0.5)
    plt.title('Сравнение статистических распределений (Real vs Fake)')
    plt.xlabel('Значение признака / бита');
    plt.ylabel('Плотность вероятности');
    plt.legend();
    plt.grid(True)
    plt.savefig("thesis_plots/2_real_vs_fake_dist.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Progression обучения (Визуализация изменения сэмплов)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(' Progression обучения: Эволюция структуры сгенерированного шифротекста')

    colors = ['#ff7f0e', '#bcbd22', '#17becf']
    for i, (epoch, sample) in enumerate(prog_samples.items()):
        axs[i].bar(range(MSG_LEN), sample, color=colors[i], alpha=0.8)
        axs[i].set_title(f'Эпоха {epoch}')
        axs[i].set_ylim(0, 1);
        axs[i].set_xlabel('Индекс бита')
        if i == 0: axs[i].set_ylabel('Вероятность Sigmoid')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("thesis_plots/3_training_progression.png", dpi=300)
    plt.close()

    # Результаты генерации (Визуализация финального битового потока)
    # Шифруем тестовую фразу
    test_key = torch.rand((1, KEY_LEN))
    test_S = torch.FloatTensor(real_raw[:SENSOR_DIM]).unsqueeze(0)
    _, final_bits = encrypt_text("IITU SENSOR", test_key, test_S)

    plt.figure(figsize=(10, 4))

    # Строки = количество букв, Столбцы = MSG_LEN (24 бита)
    test_phrase = "IITU SENSOR"
    bit_matrix = np.array(final_bits).reshape(len(test_phrase), MSG_LEN)

    plt.imshow(bit_matrix, cmap='binary', aspect='auto')

    # Добавляем подписи осей для наглядности в отчете
    plt.ylabel('Символы сообщения')
    plt.xlabel('Биты шифротекста (0 до 23)')
    plt.title('Визуализация финального сгенерированного битового потока (Raw Bits)')
    plt.axis('off')  # Убираем оси для красивого вида "шума"
    plt.savefig("thesis_plots/4_generation_results_vis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Все 4 графика сохранены в папку 'thesis_plots'. Используйте их в диссертации.")



if __name__ == "__main__":
    dataloader = load_sensor_dataset("X_train.txt")

    # Обучение и сбор данных
    l_bob, l_eve, prog, r_raw, f_raw = train_model(dataloader)

    # ВИЗУАЛИЗАЦИЯ
    visualize_all_results(l_bob, l_eve, prog, r_raw, f_raw)

    # Тестовое шифрование и экспорт
    test_key = torch.rand((1, KEY_LEN))
    test_sensor_features, = next(iter(dataloader))
    single_sensor_feature = test_sensor_features[0].unsqueeze(0)
    tokens, bits = encrypt_text("IITU SENSOR GAN", test_key, single_sensor_feature)

    with open("./export/gan_keys_raw_ascii.txt", "w") as f:
        f.write("".join(map(str, bits)))
    print("\nПроект успешно завершен.")