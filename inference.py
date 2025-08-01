import sys

from matplotlib import pyplot as plt
from skimage import measure
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from networks.segment_anything_main.segment_anything.build_sam_moe_256 import sam_model_registry as sam_moe_v2_256
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from utils_.metric import calculate_metrics, sum_metrics
from config.config import getargs
from utils_.Throat_Dataset import Throatdataset
from utils_.framework import MT_framework


def main(gpu, args, test_set):
    file_path = args.root_path
    rank = args.nr * args.gpus + gpu  # 获取当前进程
    if rank == 0:
        print(f'gpu:{gpu}')
        print("rank:", rank)
        print("world_size:", args.world_size)
        print("usable GPU", torch.cuda.device_count())

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)
    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=rank, drop_last=False)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                            sampler=test_sampler)

    model_name = "Semi_MedSAM" + str(args.label_num)
    snapshot_path = file_path + args.dataset + "/checkpoints/" + model_name
    os.makedirs(snapshot_path, exist_ok=True)

    sam_checkpoint = r''
    model_type = "vit_b"
    model = sam_moe_v2_256[model_type](gpu, args=args, num_classes=2, checkpoint=sam_checkpoint)
    model_ema = sam_moe_v2_256[model_type](gpu, args=args, num_classes=2, checkpoint=sam_checkpoint)
    model.requires_grad_(False)
    for key, value in model.image_encoder.named_parameters():
        if "Adapter" in key:
            value.requires_grad = True
    for key, value in model.mask_decoder.named_parameters():
        value.requires_grad = True

    model_ema.requires_grad_(False)
    for param in model_ema.parameters():
        param.detach_()

    net = MT_framework(args, model, model_ema, gpu)
    net = net.cuda(gpu)
    net = DDP(net, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if rank == 0:
        logging.info(str(args))

    with torch.no_grad():
        net.eval()
        metric_total = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        count_test = 0
        for i_batch, sample in enumerate(testloader):
            count_test = count_test + 1
            img_n_lb, img_w_lb = sample[0]['img_n'].cuda(gpu), sample[0]["img_w"].cuda(gpu)
            label_w = sample[0]['label_w'].cuda(gpu)
            img_item = sample[1]['id'][0]

            pred_lb, pred_ulb, pred_lb_ema, pred_ulb_ema, pseudo_label, pseudo_label_ema, _ = net(
                torch.concat((img_n_lb, img_n_lb), dim=0), torch.concat((img_w_lb, img_w_lb), dim=0),
                label_w)
            pred = torch.argmax(F.softmax(pred_lb, dim=1).detach(), dim=1, keepdim=False)
            metric = calculate_metrics(pred, label_w)
            metric_total = sum_metrics(metric_total, metric)

            save_path = file_path + args.dataset + "/visulize/pred/"
            os.makedirs(save_path, exist_ok=True)
            img_n_np = img_n_lb[0].cpu().numpy() / 255
            img_w_np = img_w_lb[0].cpu().numpy() / 255
            label_w[label_w == 255] = 0
            label_w_np = label_w[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            contours = measure.find_contours(label_w_np, level=0.5)
            contours_high_res = []
            for contour in contours:
                contours_high_res.append(contour * 4)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"Image: {img_item}", fontsize=16)
            axes[0].imshow(img_n_np)
            axes[0].set_title('NBI')
            for contour in contours_high_res:
                axes[0].plot(contour[:, 1], contour[:, 0], linewidth=2, color='lime')
            axes[0].axis('off')
            axes[1].imshow(img_w_np)
            axes[1].set_title('WLI')
            for contour in contours_high_res:
                axes[1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='lime')
            axes[1].axis('off')
            axes[2].imshow(pred_np, cmap='gray')
            axes[2].set_title('Prediction')
            for contour in contours:
                axes[2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='lime')
            axes[2].axis('off')
            save_path = os.path.join(save_path, f"comparison_{i_batch + gpu * len(testloader)}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
            print(f"Saved comparison to {save_path}")

        reduce_count_test = torch.tensor(count_test).cuda(gpu)
        reduce_dice_avg = metric_total[0].clone().detach().cuda(gpu)
        reduce_dice_0 = metric_total[1].clone().detach().cuda(gpu)
        reduce_dice_1 = metric_total[2].clone().detach().cuda(gpu)
        reduce_miou = torch.tensor(metric_total[3]).cuda(gpu)
        reduce_iou_0 = metric_total[4].clone().detach().cuda(gpu)
        reduce_iou_1 = metric_total[5].clone().detach().cuda(gpu)
        reduce_sp = torch.tensor(metric_total[6]).cuda(gpu)
        reduce_se = torch.tensor(metric_total[7]).cuda(gpu)
        dist.reduce(reduce_dice_avg, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_dice_0, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_dice_1, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_miou, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_iou_0, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_iou_1, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_sp, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_se, 0, op=dist.ReduceOp.SUM)
        dist.reduce(reduce_count_test, 0, op=dist.ReduceOp.SUM)

        if rank == 0:
            print("dive_avg", reduce_dice_avg.item() / reduce_count_test.item())
            print("dive_0", reduce_dice_0.item() / reduce_count_test.item())
            print("dive_1", reduce_dice_1.item() / reduce_count_test.item())
            print("miou", reduce_miou.item() / reduce_count_test.item())
            print("iou_0", reduce_iou_0.item() / reduce_count_test.item())
            print("iou_1", reduce_iou_1.item() / reduce_count_test.item())
            print("sp", reduce_sp.item() / reduce_count_test.item())
            print("se", reduce_se.item() / reduce_count_test.item())


if __name__ == "__main__":
    args = getargs()
    args.world_size = args.nodes * args.gpus
    args.dataset = "Throat"
    args.resize_shape = 1024
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "Throat":
        test_num = 17
        r = range(test_num)
        test_idxs = random.sample(r, test_num)
        test_path = "annotation_demo.json"
        dataset_test = Throatdataset(args, test_path, index=test_idxs, aug_type="weak", mode="demo",
                                     resize_shape=args.resize_shape)
    else:
        raise NotImplementedError

    mp.spawn(
        main,
        nprocs=args.gpus,
        args=(args, dataset_test)
    )
