# AI-Powered SDR & Communication Systems Laboratory

An interactive, real-time Software Defined Radio (SDR) and communication systems simulator built with Python. This application combines digital signal processing (DSP), machine learning, and a hardware-style Cyberpunk GUI to simulate standard analog modulation schemes (including linear and exponential methods), channel impairments, and automated signal classification.

![SDR Lab Interface](image_a69921.png)

---

## 🚀 Key Architectural Features

### 1. Digital Signal Processing (DSP) Engine
The core communication pipeline is modeled entirely in software using vectorized NumPy operations and SciPy filter topologies:
* **Linear Modulation Schemes:** Implements Amplitude Modulation (AM), Double Sideband-Suppressed Carrier (DSB-SC), and Single Sideband-Suppressed Carrier (SSB-SC). The SSB-SC module utilizes a wideband **Hilbert Transform** to shift the baseband phase by -90° for strict sideband isolation.
* **Exponential Modulation (FM):** Implements **Frequency Modulation (FM)** where the instantaneous frequency of the carrier varies linearly with the message amplitude. This is achieved by calculating the discrete time integral of the baseband signal using `np.cumsum` to track continuous phase accumulation.
* **Demodulation Topologies:** Features an asynchronous **Envelope Detector** (rectification followed by a 5th-order Butterworth low-pass filter) for AM, a synchronous **Coherent Detector** for DSB/SSB, and a **Frequency Discriminator/Quadrature Demodulator** loop to extract instantaneous phase deviations for FM recovery.

### 2. Multi-Channel Visualizer (Real-Time UI)
Built on `PyQt5` and optimized via `pyqtgraph` to handle dense arrays smoothly without blocking the asynchronous audio thread:
* **Time-Domain Oscilloscope:** Live plots tracking instantaneous voltages of the Message, Modulated (TX), and Channel-Impaired Received (RX) signals over N = 4096 samples. For FM, this beautifully visualizes variations in wave crowding (frequency compression and expansion).
* **Frequency-Domain Analyzer:** Real-time Fast Fourier Transform (FFT) computing the magnitude spectrum converted into logarithmic decibel units (dBV). This perfectly demonstrates **Carson's Bandwidth Rule** and sideband power distribution as the modulation index varies.
* **Waterfall Display:** A rolling 2D spectrogram mapping spectral density over time, capturing live bandwidth and frequency variations.

### 3. Edge-AI Signal & Channel Classification
Features an embedded **Random Forest Classifier** trained dynamically on initialization to classify incoming waveforms based on Statistical Feature Extraction:
* **Feature Vectors:** Extracted statistics include sample mean, standard deviation, and peak amplitude.
* **Inference Classes:** Classifies waveforms in real-time as `SINE`, `SQUARE`, or `NOISE`.
* **Channel Quality Estimator:** Monitors configured Additive White Gaussian Noise (AWGN) levels to dynamically bucket communication link viability into `EXCELLENT`, `GOOD`, `POOR`, or `VERY NOISY`.

### 4. Live Channel Physics Simulation
Simulates real-world communication channel degradation before the receiver stage:
* **Multi-path Fading:** Simulates a two-ray channel model by introducing a low-frequency time-varying fading envelope (2 Hz sinusoidal fade) and a localized time-delayed multipath echo via vector rotation/rolling.
* **AWGN Channel:** Seamlessly injects customizable Gaussian noise levels into the high-frequency carrier wave.

---

## 🛠️ Technical Stack
* **Language:** Python 3
* **GUI Framework:** PyQt5 (Fusion Style Sheet)
* **Graphics & Plotting:** pyqtgraph
* **DSP Core:** NumPy, SciPy (Signal Module)
* **Machine Learning:** Scikit-Learn
* **Audio I/O:** SoundDevice (Asynchronous Stream Processing)

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI-Powered-SDR-Laboratory.git](https://github.com/YOUR_USERNAME/AI-Powered-SDR-Laboratory.git)
   cd AI-Powered-SDR-Laboratory
