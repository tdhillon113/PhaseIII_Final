from binary import run_binary
from multiclass import run_multiclass

def main():
    print("\n" + "="*60)
    print("RUNNING BINARY CLASSIFICATION (WINE DATASET)")
    print("="*60)
    run_binary()

    print("\n" + "="*60)
    print("RUNNING MULTICLASS CLASSIFICATION (BREAST CANCER DATASET)")
    print("="*60)
    run_multiclass()

    print("\n" + "="*60)
    print("ALL PIPELINES COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
