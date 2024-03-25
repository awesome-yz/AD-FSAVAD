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
    
class TestDataset(Dataset):
    def __init__(self, data, device, transform=None):
        self.img, self.label = data[0], data[1]
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        all_img = []
        all_lbl = []
        for task in self.img[index]:
            task_list = []
            for frame_sequence in task:
                if len(frame_sequence) == 4:
                    frames = []
                    img = []
                    gt = []
                    for im in frame_sequence[:3]:
                        im_opened = Image.open(im).convert('RGB')
                        if self.transform is not None:
                            img.append(self.transform(im_opened).to(self.device))
                    gt_path = frame_sequence[3]
                    gt_img = Image.open(gt_path).convert('RGB')
                    if self.transform is not None:
                        gt_transformed = self.transform(gt_img).to(self.device)
                        gt.append(gt_transformed)
                    frames.append(img)
                    frames.append(gt)
                    task_list.append(frames)
            all_img.append(task_list)
        for task in self.label[index]:
            task_list = []
            for label_sequence in task:
                labels = []
                if len(label_sequence) == 4:
                    labels.append(label_sequence[-1])
                else:
                    labels.append(label_sequence[0])
                task_list.append(labels)
            all_lbl.append(task_list)
        return all_img, all_lbl
    
    def __len__(self):
        return len(self.img)