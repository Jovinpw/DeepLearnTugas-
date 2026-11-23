# Studi Komparatif Empiris Model Generatif & Pembuktian Batasan MLP Classifier

Repositori kode eksperimen untuk **Tugas Take-Home Deep Learning (D6421) — Sesi 07: Generative Models terintegrasi Sesi 06 (Autoencoders)** pada dataset benchmark **Fashion-MNIST**.

## Ringkasan Arsitektur yang Dievaluasi

1. **Discriminative MLP Classifier & Frozen Latent Decoder** (`784 -> 256 -> 64 -> 10`, pembuktian 5 tahap ketidakmampuan bobot $W^T$ dan ruang laten $z_{\text{MLP}} \in \mathbb{R}^{64}$ untuk sintesis citra).
2. **Vanilla Autoencoder** (Deterministik, bottleneck padat $z_{\text{AE}} \in \mathbb{R}^{64}$, objektif MSE).
3. **Sparse Autoencoder** (Deterministik, bottleneck $h \in \mathbb{R}^{64}$ dengan regularisasi kejarangan $L_1$, $\lambda = 1.2 \times 10^{-3}$).
4. **Variational Autoencoder / VAE** (Probabilistik, ruang laten kontinu $\mu, \log\sigma^2 \in \mathbb{R}^{64}$, objektif Negative ELBO).
5. **Vanilla GAN** (Arsitektur MLP adversarial, prior noise $z \sim \mathcal{N}(0, I_{64})$).
6. **Deep Convolutional GAN / DCGAN** (Arsitektur transposed & strided convolution dengan Batch Normalization, prior noise $z \sim \mathcal{N}(0, I_{100})$).
7. **Denoising Diffusion Probabilistic Model / DDPM** (Arsitektur 2D U-Net terkondisi waktu *Sinusoidal Time Embedding*, jadwal difusi linier $T = 400$).

## Struktur Direktori

```text
.
├── notebook_studi_komparatif_jovin.ipynb   # Notebook utama eksperimen, evaluasi metrik, dan visualisasi
├── requirements.txt                        # Daftar pustaka Python (PyTorch, Torchvision, SciPy, Matplotlib)
├── README.md                               # Dokumentasi dan panduan eksekusi
└── outputs/                                # Direktori luaran metrik JSON, skema arsitektur, dan 5 figur wajib
    ├── skema_1_autoencoder_mlp.png
    ├── skema_2_adversarial_gan.png
    ├── skema_3_ddpm_unet.png
    ├── fig1_rekonstruksi_komparatif.png
    ├── fig2_sampel_generatif_4x4.png
    ├── fig3_pembuktian_mlp_classifier.png
    ├── fig4_interpolasi_ruang_laten.png
    ├── fig5_kurva_loss_dan_difusi.png
    ├── experiment_metrics.json
    └── training_logs.json
```

## Standar Reproduktibilitas & Petunjuk Eksekusi

- **Dataset**: `torchvision.datasets.FashionMNIST` (70.000 citra $28 \times 28$ grayscale), dibagi secara seragam menjadi **80% Train Set (56.000)**, **10% Validation Set (7.000)**, dan **10% Test Set (7.000)**.
- **Random Seed & Batch Size**: `SEED = 42` (`torch.manual_seed(42)`, `np.random.seed(42)`), `BATCH_SIZE = 256`.
- **Cara Menjalankan**:
  1. Instal dependensi pustaka:
     ```bash
     pip install -r requirements.txt
     ```
  2. Jalankan seluruh sel secara berurutan pada `notebook_studi_komparatif_jovin.ipynb`. Seluruh hasil metrik dan figur resolusi tinggi (300 DPI) akan otomatis tersimpan di dalam folder `outputs/`.
