from configuration import N_CLASSES, DEVICE
import torch


# 평가 메트릭인 mIoU 계산 과정
def compute_iou_tensor(pred_map, target_map):
    intersection = (pred_map & target_map).sum(dim=(1, 2)).float()
    union = (pred_map | target_map).sum(dim=(1, 2)).float()
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return iou

def eval(model, train_loader):
    model.eval()
    total_iou = torch.zeros(N_CLASSES).to(DEVICE)
    total_samples = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 클래스별 mIoU 계산
            for cls in range(N_CLASSES):
                pred_map = (preds == cls)
                target_map = (targets == cls)

                iou = compute_iou_tensor(pred_map, target_map)
                total_iou[cls] += iou.sum()

            total_samples += images.size(0)

    mean_iou_per_class = total_iou / total_samples
    mIoU = mean_iou_per_class.mean().item()

    print(f"[EVAL] mIoU: {mIoU:.4f}")
    return mIoU