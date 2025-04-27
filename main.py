from prepare_datasets import prepare_all_datasets
from train import train_with_transfer_learning

def main():
    prepare_all_datasets()
    model, results=train_with_transfer_learning()
    predictions=model.val(data="VOC.yaml")

if __name__ == "__main__":
    main()
