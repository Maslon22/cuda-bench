import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11


def plot_time(df, title, outfile):
    plt.figure()
    for (device, variant), g in df.groupby(["device", "variant"]):
        plt.plot(g["size"], g["time_ms"], marker="o", label=f"{device} {variant}")
    plt.xlabel("Rozmiar problemu")
    plt.ylabel("Czas [ms]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)
    plt.close()


def plot_gflops(df, title, ylabel, outfile):
    plt.figure()
    for (device, variant), g in df.groupby(["device", "variant"]):
        plt.plot(g["size"], g["gflops"], marker="o", label=f"{device} {variant}")
    plt.xlabel("Rozmiar problemu")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)
    plt.close()


def plot_speedup(df, title, outfile):
    gpu = df[df["device"] == "GPU"]
    plt.figure()
    for variant, g in gpu.groupby("variant"):
        plt.plot(g["size"], g["speedup"], marker="o", label=f"GPU {variant}")
    plt.xlabel("Rozmiar problemu")
    plt.ylabel("Speedup (CPU / GPU)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)
    plt.close()


# ================== MATMUL ==================
if os.path.exists("results_matmul.csv"):
    df = pd.read_csv("results_matmul.csv")
    plot_time(df, "Mnożenie macierzy – czas wykonania", "matmul_time.png")
    plot_gflops(df, "Mnożenie macierzy – wydajność", "GFLOPS", "matmul_gflops.png")
    plot_speedup(df, "Mnożenie macierzy – speedup GPU", "matmul_speedup.png")

# ================== FFT ==================
if os.path.exists("results_fft.csv"):
    df = pd.read_csv("results_fft.csv")
    plot_time(df, "FFT – czas wykonania", "fft_time.png")
    plot_gflops(df, "FFT – wydajność", "GFLOPS", "fft_gflops.png")
    plot_speedup(df, "FFT – speedup GPU", "fft_speedup.png")

# ================== MONTE CARLO ==================
if os.path.exists("results_montecarlo.csv"):
    df = pd.read_csv("results_montecarlo.csv")
    plot_time(df, "Monte Carlo – czas wykonania", "monte_time.png")
    plot_gflops(
        df,
        "Monte Carlo – przepustowość",
        "Samples / ms",
        "monte_throughput.png",
    )
    plot_speedup(df, "Monte Carlo – speedup GPU", "monte_speedup.png")

# ================== CONV2D ==================
if os.path.exists("results_conv2d.csv"):
    df = pd.read_csv("results_conv2d.csv")
    plot_time(df, "Konwolucja 2D – czas wykonania", "conv_time.png")
    plot_gflops(df, "Konwolucja 2D – wydajność", "GFLOPS", "conv_gflops.png")
    plot_speedup(df, "Konwolucja 2D – speedup GPU", "conv_speedup.png")

print("Wykresy zapisane.")
