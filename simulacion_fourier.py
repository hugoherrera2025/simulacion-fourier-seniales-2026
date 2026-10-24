import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SIMULACIÓN Y ANÁLISIS DE SEÑALES CON FOURIER
# ==========================================

# 1. Definición del dominio del tiempo
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.arange(-2, 2, 1/fs)

# --- Señales Elementales ---
# A) Pulso rectangular (duración de 1 segundo, de -0.5 a 0.5)
pulso_rect = np.where(np.abs(t) <= 0.5, 1.0, 0.0)

# B) Función escalón unitario
escalon = np.where(t >= 0, 1.0, 0.0)

# C) Señal senoidal (frecuencia de 5 Hz)
f_sen = 5
senoidal = np.sin(2 * np.pi * f_sen * t)

# 2. Cálculo de la Transformada de Fourier usando FFT
# Frecuencias asociadas a la FFT
freqs = np.fft.fftfreq(len(t), 1/fs)

# Transformadas de Fourier
F_rect = np.fft.fft(pulso_rect) / fs
F_esc = np.fft.fft(escalon) / fs
F_sen = np.fft.fft(senoidal) / fs

# Reordenar para que la frecuencia cero esté en el centro
freqs_centradas = np.fft.fftshift(freqs)
F_rect_cent = np.fft.fftshift(F_rect)
F_esc_cent = np.fft.fftshift(F_esc)
F_sen_cent = np.fft.fftshift(F_sen)

# 3. Visualización de Resultados (Gráficas)
plt.figure(figsize=(14, 10))

# --- Pulso Rectangular ---
plt.subplot(3, 2, 1)
plt.plot(t, pulso_rect, 'b')
plt.title('Dominio del Tiempo: Pulso Rectangular')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(freqs_centradas, np.abs(F_rect_cent), 'r')
plt.title('Espectro de Frecuencia (Magnitud): SINC')
plt.xlim(-20, 20)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X(f)|')
plt.grid(True)

# --- Función Escalón ---
plt.subplot(3, 2, 3)
plt.plot(t, escalon, 'b')
plt.title('Dominio del Tiempo: Función Escalón')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(freqs_centradas, np.abs(F_esc_cent), 'r')
plt.title('Espectro de Frecuencia: Escalón')
plt.xlim(-20, 20)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X(f)|')
plt.grid(True)

# --- Señal Senoidal ---
plt.subplot(3, 2, 5)
plt.plot(t, senoidal, 'b')
plt.title('Dominio del Tiempo: Senoidal (5 Hz)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(freqs_centradas, np.abs(F_sen_cent), 'r')
plt.title('Espectro de Frecuencia: Impulsos en ±5 Hz')
plt.xlim(-15, 15)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X(f)|')
plt.grid(True)

plt.tight_layout()
plt.show()

print("¡Simulación completada con éxito! Las gráficas se han generado.")
