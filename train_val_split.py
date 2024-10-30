    
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    text_file_to_split = Path('/tmp/oxford-pet/annotations/trainval.txt')
    random_seed = 1994
     
    split_ratio = 0.2

    val_txt_path = text_file_to_split.parent / "val.txt"
    train_txt_path = text_file_to_split.parent / "train.txt"

    with open(str(text_file_to_split), "r") as f:
        data = f.readlines()
    
    n = len(data)
    val_size = int(n * split_ratio)
    
    np_random = np.random.RandomState(random_seed)
    np_random.shuffle(data)

    val_data = data[:val_size]
    train_data = data[val_size:]

    with open(str(val_txt_path), "w") as f:
        f.writelines(val_data)
    with open(str(train_txt_path), "w") as f:
        f.writelines(train_data)
    
    print(f"Data split into train and val with ratio {split_ratio}")
    print(f"Train data saved in {train_txt_path}")
    print(f"Val data saved in {val_txt_path}")