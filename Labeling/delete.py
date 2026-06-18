import fiftyone as fo

DATASET_NAME = "weight"

if DATASET_NAME in fo.list_datasets():
    fo.delete_dataset(DATASET_NAME)
    print("Dataset gelöscht!")
else:
    print("Dataset existiert nicht.")
