import torch
import torchvision

def test_cuda():
    print("====================================")
    print("🔍 CEK STATUS CUDA DI PYTORCH")
    print("====================================\n")

    # Cek apakah PyTorch mendeteksi CUDA
    print(f"CUDA available       : {torch.cuda.is_available()}")
    print(f"CUDA device count    : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"CUDA current device  : {torch.cuda.current_device()}")
        print(f"CUDA device name     : {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Cek versi torch & torchvision
    print(f"\nPyTorch version      : {torch.__version__}")
    print(f"Torchvision version  : {torchvision.__version__}")

    # Cek kemampuan menjalankan tensor di GPU
    if torch.cuda.is_available():
        try:
            x = torch.rand(3, 3).cuda()
            print("\nTensor di GPU berhasil dibuat ✔")
            print(x)
        except Exception as e:
            print("\nGagal membuat tensor di GPU ❌")
            print(e)
    else:
        print("\nCUDA tidak tersedia ❌")

    print("\n====================================")
    print("🔧 Jika CUDA tidak terdeteksi, cek:")
    print("1. Environment Conda yang aktif")
    print("2. Instalasi pytorch-cuda=11.8")
    print("3. NVIDIA driver harus terbaru")
    print("====================================")

if __name__ == "__main__":
    test_cuda()
