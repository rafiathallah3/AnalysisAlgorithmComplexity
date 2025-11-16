import matplotlib.pyplot as plt
import time
import sys

sys.setrecursionlimit(10000)

def pangkat_rekursif_naif(x, n):
    """
    Kalkulasi x^n menggunakan rekursif naif.
    Time Complexity: O(n)
    """
    if n == 0:
        return 1
    else:
        return x * pangkat_rekursif_naif(x, n - 1)

def pangkat_iterasi_naif(x, n):
    """
    Kalkulasi x^n menggunakan loop iterasi sederhana.
    Time Complexity: O(n)
    """
    result = 1
    for _ in range(n):
        result *= x
    return result

def pangkat_rekursif_cepat(x, n):
    """
    Kalkulasi x^n menggunakan eksponensiasi cepat (pangkatkan dengan kuadrat).
    Ini tu algoritma yang namanya "divide and conquer".
    Time Complexity: O(log n)
    """
    if n == 0:
        return 1
    
    bagi = pangkat_rekursif_cepat(x, n // 2)
    
    if n % 2 == 0:
        return bagi * bagi
    else:
        return x * bagi * bagi

if __name__ == "__main__":
    angka = 2
    
    semua_n = range(1, 10001, 20)
    
    waktu_rekursif_naif = []
    waktu_iterasi_naif = []
    waktu_rekursif_cepat = []

    print(f"Dimulai dengan angka = {angka}, dan n dari 1 sampe {semua_n.stop}...")
    
    for n in semua_n:
        waktu_mulai = time.perf_counter()
        pangkat_rekursif_naif(angka, n)
        waktu_selesai = time.perf_counter()
        waktu_rekursif_naif.append(waktu_selesai - waktu_mulai)
        
        waktu_mulai = time.perf_counter()
        pangkat_iterasi_naif(angka, n)
        waktu_selesai = time.perf_counter()
        waktu_iterasi_naif.append(waktu_selesai - waktu_mulai)
        
        waktu_mulai = time.perf_counter()
        pangkat_rekursif_cepat(angka, n)
        waktu_selesai = time.perf_counter()
        waktu_rekursif_cepat.append(waktu_selesai - waktu_mulai)

    print("Sudah selesai!! Buatin plot...")
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(semua_n, waktu_rekursif_naif, label="Recursive (Naive) - O(n)")
    plt.plot(semua_n, waktu_iterasi_naif, label="Iterative (Naive) - O(n)")
    plt.plot(semua_n, waktu_rekursif_cepat, label="Recursive (Fast) - O(log n)")
    
    plt.xlabel("Input Size 'n' (the exponent)")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Algorithm Complexity Comparison: Calculating {angka}^n")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


    plt.show()