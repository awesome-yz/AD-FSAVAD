from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, img, transform, device, rgb=True):
        self.img= img
        self.transform=transform
        self.rgb = rgb
        self.device=device

    def __getitem__(self, index):
        epoch = []
        for task in self.img[index]:
            task_list = []
            for frame_sequence in task:
                seq = []
                imgs = []
                gt = []
                for img in frame_sequence[:3]:
                    img_convert = Image.open(img).convert('RGB')
                    if self.transform is not None:
                        img_transformed = self.transform(img_convert).to(device=self.device)
                        imgs.append(img_transformed)
                gt_path = frame_sequence[3]
                gt_img = Image.open(gt_path).convert('RGB')
                if self.transform is not None:
                    gt_transformed = self.transform(gt_img).to(device=self.device)
                    gt.append(gt_transformed)

                seq.append(imgs)
                seq.append(gt)
                task_list.append(seq)
            epoch.append(task_list)
        return epoch

    def __len__(self):
        return len(self.img)