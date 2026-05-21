import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import hashlib
import time
import os


os.makedirs("export", exist_ok=True)

# ГИПЕРПАРАМЕТРЫ
MSG_LEN = 24
KEY_LEN = 24
SENSOR_DIM = 561
BATCH_SIZE = 128
NUM_EPOCHS = 50

INPUT_DIM = MSG_LEN + KEY_LEN + SENSOR_DIM

# МОДЕЛИ
class CryptoGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
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
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ИНИЦИАЛИЗАЦИЯ
alice = CryptoGenerator(INPUT_DIM, MSG_LEN)
bob = CryptoGenerator(INPUT_DIM, MSG_LEN)
eve = CryptoDiscriminator(MSG_LEN, MSG_LEN)

opt_ab = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=0.0008)
opt_e = optim.Adam(eve.parameters(), lr=0.0008)

loss_fn = nn.MSELoss()

# DATASET
def load_sensor_dataset():
    data = np.random.uniform(-1, 1, (5000, SENSOR_DIM))
    dataset = TensorDataset(torch.FloatTensor(data))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# TRAIN STEP
def train(dataloader):
    alice.train(); bob.train(); eve.train()

    for epoch in range(NUM_EPOCHS):
        for sensor_data, in dataloader:

            P = torch.randint(0, 2, (BATCH_SIZE, MSG_LEN)).float()
            K = torch.randint(0, 2, (BATCH_SIZE, KEY_LEN)).float()
            S = sensor_data

            # -------------------
            # Eve
            # -------------------
            opt_e.zero_grad()

            C = alice(torch.cat([P, K, S], dim=1))
            P_eve = eve(C.detach())

            loss_e = loss_fn(P_eve, P)
            loss_e.backward()
            opt_e.step()

            # -------------------
            # Alice + Bob
            # -------------------
            opt_ab.zero_grad()

            C = alice(torch.cat([P, K, S], dim=1))
            P_bob = bob(torch.cat([C, K, S], dim=1))

            loss_b = loss_fn(P_bob, P)
            loss_adv = loss_fn(eve(C), P)

            loss_ab = loss_b + (1 - loss_adv).pow(2)
            loss_ab.backward()

            opt_ab.step()

        print(f"Epoch {epoch} | Bob loss {loss_b.item():.4f} | Eve loss {loss_e.item():.4f}")

# ENCRYPT FUNCTION
def encrypt(message, key, sensor):
    alice.eval()

    out = []
    for ch in message:
        bits = torch.tensor([int(b) for b in bin(ord(ch))[2:].zfill(MSG_LEN)]).float()

        inp = torch.cat([bits.unsqueeze(0), key, sensor], dim=1)

        with torch.no_grad():
            g = alice(inp)[0]

        g_bits = (g > 0).int().numpy()

        out.append(g_bits)

    return np.array(out)

def save_bits_for_nist(bits, filename="export/nist_bits.txt"):

    os.makedirs("export", exist_ok=True)

    # приводим к 1D массиву
    flat_bits = np.array(bits).astype(int).flatten()

    # сохраняем как строку 0/1
    bit_string = "".join(map(str, flat_bits.tolist()))

    with open(filename, "w") as f:
        f.write(bit_string)

    print(f"NIST bitstream saved: {filename} | length = {len(bit_string)}")


def save_models():
    torch.save(alice.state_dict(), "export/alice.pth")
    print("Saved .pth models")


def export_torchscript():
    dummy = torch.randn(1, INPUT_DIM)

    traced = torch.jit.trace(alice, dummy)
    traced.save("export/alice_mobile.pt")

    print("Saved TorchScript model")

# MAIN
if __name__ == "__main__":
    loader = load_sensor_dataset()
    train(loader)
    save_models()
    export_torchscript()

    # test
    key = torch.rand((1, KEY_LEN))
    sensor = torch.rand((1, SENSOR_DIM))

    print(encrypt("HELLO", key, sensor))
    save_bits_for_nist(encrypt("HELLO", key, sensor))