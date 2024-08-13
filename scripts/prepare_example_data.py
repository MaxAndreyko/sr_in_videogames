from pathlib import Path
from torchvision import transforms
from PIL import Image


def main(img_url, dataset_name, size=10):
    root_dir = Path(f"datasets/{dataset_name}")
    lr_dir = root_dir.joinpath("lr")
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir = root_dir.joinpath("hr")
    hr_dir.mkdir(parents=True, exist_ok=True)
    hr_img_raw = Image.open(img_url).convert("RGB")

    # случайным образом получим HR изображения размера (256, 256)
    for i in range(size):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop((1024, 1024))
            ]
        )

        # получим LR изображение размера (64, 64) при помощи простого resize
        resize_transform = transforms.Resize((256, 256))

        # tensor to PIL
        tensor2pil = transforms.ToPILImage()

        hr_img = transform(hr_img_raw)
        lr_img = resize_transform(hr_img)
        hr_img = tensor2pil(hr_img)
        lr_img = tensor2pil(lr_img)
        hr_img.save(hr_dir / f"{str(i)}.png")
        lr_img.save(lr_dir / f"{str(i)}.png")


if __name__ == '__main__':
    img_urls = ["datasets/hr.png"]
    datasets = ['ExampleDataset']
    for img_url, dataset in zip(img_urls, datasets):
        main(img_url, dataset)
