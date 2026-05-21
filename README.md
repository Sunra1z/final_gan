# final_gan
Neural Entropy Conditioned Cryptographic GAN

A neural adversarial cryptographic system that generates secure bitstreams using sensor-based entropy and GAN-style training (Alice–Bob–Eve architecture).

##  Overview

This project implements a neural cryptographic framework inspired by adversarial learning. The system learns to:

- Encrypt messages using a learned generator (Alice)
- Decrypt messages using a cooperative model (Bob)
- Resist adversarial attacks (Eve)
- Enhance randomness using real-world sensor data (UCI HAR dataset)

The final output is a **pseudo-random bitstream** suitable for statistical randomness testing (e.g., NIST SP 800-22).

---

## 🧠 Architecture

### 🔹 Alice (Generator / Encryptor)
- Input: `Message bits + Key + Sensor features`
- Output: `24-bit ciphertext`
- Layers:
  - Linear → ReLU → Dropout → Linear → Tanh

---

### 🔹 Bob (Decryptor)
- Input: `Ciphertext + Key + Sensor features`
- Output: reconstructed message bits
- Same architecture as Alice

---

### 🔹 Eve (Adversary)
- Input: `Ciphertext only`
- Goal: reconstruct original message without key
- Used to enforce cryptographic robustness

---

## ⚙️ Training Objective

Adversarial training follows:

Bob minimizes reconstruction loss
---

Eve minimizes attack loss
---

Alice minimizes combined adversarial loss
---

## 📊 Dataset

### UCI Human Activity Recognition (HAR)

- 561 sensor features
- Normalized range: [-1, 1]
- Source: smartphone accelerometer & gyroscope signals

Used as a **physical entropy source** to improve randomness quality.

---

## 🔐 Key Features

- Adversarial neural cryptography (Alice–Bob–Eve)
- Sensor-based entropy injection
- Residual noise augmentation
- XOR-based stochastic bit refinement
- Neural pseudo-random bit generator (PRNG)

---

## 📈 Outputs

The system generates:

- Encrypted bitstreams (24 bits per symbol)
- Large-scale binary sequences (up to millions of bits)
- Statistical randomness data for NIST testing
