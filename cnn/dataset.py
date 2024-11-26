import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug: list = []) -> None:
        self.dataset: list[tuple[str, np.ndarray]] = []
        self.root_dir = root_dir

        self.__add_dataset__("bear",        [1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("camel",       [0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("giraffe",     [0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("horse",       [0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("kangaroo",    [0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("koala",       [0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("zebra",       [0, 0, 0, 0, 0, 0, 1])

        post_processing = [
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        self.augmentation = transforms.Compose(
            [transforms.Resize((150, 150))] +
            aug +
            post_processing
        )

    def __add_dataset__(self, dir_name: str, class_label: list[int]) -> None:
        full_path = os.path.join(self.root_dir, dir_name)
        if not os.path.exists(full_path):
            print(f"Warning: Directory {full_path} does not exist.")
            return

        label = np.array(class_label)
        for fname in os.listdir(full_path):
            fpath = os.path.join(full_path, fname)
            if fpath.lower().endswith(('jpg', 'png')):
                self.dataset.append((fpath, label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)

        image = (image - image.min()) / (image.max() - image.min())
        label = torch.Tensor(label)

        return image, label


class SimpleMediaVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str) -> None:
        self.dataset: list[tuple[list[str], np.ndarray]] = []
        self.root_dir = root_dir

        self.__add_dataset__("bear",        [1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("camel",       [0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("giraffe",     [0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("horse",       [0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("kangaroo",    [0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("koala",       [0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("zebra",       [0, 0, 0, 0, 0, 0, 1])

        self.augmentation = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

    def __add_dataset__(self, class_name: str, class_label: list[int]) -> None:
        full_path = os.path.join(self.root_dir, class_name)
        if not os.path.exists(full_path):
            print(f"Warning: Directory {full_path} does not exist.")
            return

        label = np.array(class_label)
        self.__add_dir__(full_path, label, skip_rate=6, size_length=15)

    def __add_dir__(self, dir_path: str, cls_label: np.ndarray, skip_rate=6, size_length=15) -> None:
        files = sorted(
            [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.lower().endswith(('jpg', 'png'))]
        )

        if len(files) < size_length:
            print(f"Skipping {dir_path}, not enough files ({len(files)})")
            return

        sampled_files = [files[i] for i in range(0, len(files), skip_rate)]
        for index in range(size_length, len(sampled_files)):
            sequence = sampled_files[index - size_length:index]
            if len(sequence) == size_length:
                self.dataset.append((sequence, cls_label))
        print(f"Added {len(self.dataset)} samples from {dir_path}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __process_frame__(self, fpath: str) -> np.ndarray:
        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)
        return image[0]

    def __getitem__(self, index: int):
        path_seq, label = self.dataset[index]

        arr_x = [self.__process_frame__(x) for x in path_seq]
        arr_x = np.stack(arr_x, axis=0)

        label = torch.Tensor(label)

        return arr_x, label


if __name__ == "__main__":
    print("Testing Datasets")

    # Test the video dataset
    video_dataset = SimpleMediaVideoDataset('./dataset/val')
    print(f"Total samples in video dataset: {len(video_dataset)}")
    video_loader = torch.utils.data.DataLoader(video_dataset, batch_size=4)
    for x, y in video_loader:
        print(x.shape)
        print(y.shape)
        break

    # Test the single image dataset
    image_dataset = SimpleTorchDataset('./dataset/train')
    print(f"Total samples in image dataset: {len(image_dataset)}")
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=4)
    for x, y in image_loader:
        print(x.shape)
        print(y.shape)
        break
