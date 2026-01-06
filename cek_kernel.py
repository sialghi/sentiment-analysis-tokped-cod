import joblib

# 1. Load modelnya
model = joblib.load('model_svm_tokopedia.pkl')

# 2. Print jenis kernelnya
print(f"Jenis Kernel: {model.kernel}")

# 3. (Opsional) Lihat semua parameter
print("Detail Model:", model.get_params())