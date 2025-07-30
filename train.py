import sys

from torch import nn
from tqdm import tqdm
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
from utils_.metrics import Dice_and_ce_loss
from utils_.metric import calculate_metrics, sum_metrics
from config.config_sam_ddp import getargs
from utils_.Throat_Dataset import Throatdataset
from utils_.framework import MT_framework

file_path = r""  # todo


def main(gpu, args, train_lb, train_ulb, test_set):
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
    epoch_num = args.epoch_num
    train_sampler_lb = DistributedSampler(train_lb, num_replicas=args.world_size, rank=rank, drop_last=True)
    trainloader_lb = DataLoader(train_lb, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False,
                                sampler=train_sampler_lb)
    train_sampler_ulb = DistributedSampler(train_ulb, num_replicas=args.world_size, rank=rank, drop_last=True)
    trainloader_ulb = DataLoader(train_ulb, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False,
                                 sampler=train_sampler_ulb)
    print(f"len_trainlb:{len(trainloader_lb)}, len_trainulb:{len(trainloader_ulb)}")

    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=rank, drop_last=True)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                            sampler=test_sampler)

    model_name = "Semi_MedSAM" + str(args.label_num)
    snapshot_path = file_path + args.dataset + "/checkpoints/" + model_name
    save_path = file_path + args.dataset + "/visulize/pred/"
    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    sam_checkpoint = r""  # SAM pretrained checkpoint
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

    criterion_l = Dice_and_ce_loss().cuda(gpu)
    criterion_u = nn.CrossEntropyLoss(reduction='mean', ignore_index=255).cuda(gpu)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.base_lr, momentum=0.9,
                                weight_decay=0.0001)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if rank == 0:
        logging.info(str(args))
        logging.info("{} itertations per epoch".format(len(trainloader_lb)))

    dice_avg_best, miou_best = 0.0, 0.0
    iter_num = 0
    for epoch_num in tqdm(range(epoch_num), ncols=70):
        train_sampler_lb.set_epoch(epoch_num)
        train_sampler_ulb.set_epoch(epoch_num)
        net.train()
        loss_total, loss_sup_total, loss_usup_total, count_train = 0.0, 0.0, 0.0, 0  # train statistic
        loader = zip(trainloader_lb, trainloader_ulb)
        for i, (sample_lb, sample_ulb) in enumerate(loader):
            iter_num = iter_num + 1
            img_n_lb, img_w_lb, label_w = sample_lb[0]['img_n'].cuda(gpu), sample_lb[0]["img_w"].cuda(gpu), \
                sample_lb[0]['label_w'].cuda(gpu)
            img_n_ulb, img_w_ulb, label_w_ulb = sample_ulb[0]['img_n'].cuda(gpu), sample_ulb[0][
                "img_w"].cuda(gpu), sample_ulb[0]["label_w"].cuda(gpu)

            optimizer.zero_grad()

            pred_lb, pred_ulb, pred_lb_ema, pred_ulb_ema, pseudo_label, pseudo_label_ema, aux_loss = net(
                torch.concat((img_n_lb, img_n_ulb), dim=0), torch.concat((img_w_lb, img_w_ulb), dim=0),
                label_w_ulb)

            # supervised_loss
            dice_x, CE_loss_x, loss_x = criterion_l(pred_lb, label_w)
            loss_sup = loss_x

            # unsupervised_loss
            loss_usup = criterion_u(pred_ulb, pseudo_label_ema)

            # total_loss
            loss = torch.nansum(torch.stack(
                [loss_sup, args.consistency * net.module.get_current_consistency_weight(epoch_num) * loss_usup,
                 aux_loss]))

            loss.backward()
            optimizer.step()

            # update EMA Model
            net.module.update_ema(alpha=0.9, global_step=iter_num)

            count_train = count_train + 1
            loss_sup_total = loss_sup_total + loss_sup
            loss_usup_total = loss_usup_total + loss_usup
            loss_total = loss_total + loss

        loss_sup_tensor = loss_sup_total.cuda(gpu)
        loss_usup_tensor = loss_usup_total.cuda(gpu)
        loss_total_tensor = loss_total.cuda(gpu)
        count_train_tensor = torch.tensor(count_train).cuda(gpu)

        dist.reduce(loss_sup_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(loss_usup_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(loss_total_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(count_train_tensor, 0, op=dist.ReduceOp.SUM)
        count_train_tensor = count_train_tensor * args.batch_size
        if rank == 0:
            print(
                f"loss_sup:{loss_sup_tensor.item() / count_train_tensor.item()}, loss_usup:{loss_usup_tensor.item() / count_train_tensor.item()}, loss_total:{loss_total_tensor.item() / count_train_tensor.item()}, Epoch {epoch_num}: Learning Rate = {optimizer.param_groups[0]['lr']:.3e}\n")

        if (epoch_num + 1) % 10 == 0:
            with torch.no_grad():
                net.eval()
                metric_total = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                count_test = 0
                for i_batch, sample in enumerate(testloader):
                    count_test = count_test + 1
                    img_n_lb, img_w_lb = sample[0]['img_n'].cuda(gpu), sample[0]["img_w"].cuda(gpu)
                    label_w = sample[0]['label_w'].cuda(gpu)

                    pred_lb, pred_ulb, pred_lb_ema, pred_ulb_ema, pseudo_label, pseudo_label_ema, _ = net(
                        torch.concat((img_n_lb, img_n_lb), dim=0), torch.concat((img_w_lb, img_w_lb), dim=0),
                        label_w)
                    pred = torch.argmax(F.softmax(pred_lb, dim=1).detach(), dim=1, keepdim=False)
                    metric = calculate_metrics(pred, label_w)
                    metric_total = sum_metrics(metric_total, metric)

                reduce_count_test = torch.tensor(count_test).cuda(gpu)
                reduce_dice_avg = torch.tensor(metric_total[0]).cuda(gpu)
                reduce_dice_0 = torch.tensor(metric_total[1]).cuda(gpu)
                reduce_dice_1 = torch.tensor(metric_total[2]).cuda(gpu)
                reduce_miou = torch.tensor(metric_total[3]).cuda(gpu)
                reduce_iou_0 = torch.tensor(metric_total[4]).cuda(gpu)
                reduce_iou_1 = torch.tensor(metric_total[5]).cuda(gpu)
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

                    if reduce_dice_1 / count_test > dice_avg_best:
                        dice_avg_best = reduce_dice_1 / count_test
                        save_mode_path = os.path.join(snapshot_path, '_dice_best' + '.pth')
                        torch.save(net.module.model.state_dict(), save_mode_path)
                    if reduce_iou_1 / count_test > miou_best:
                        miou_best = reduce_iou_1 / count_test
                        save_mode_path = os.path.join(snapshot_path, '_mIou_best' + '.pth')
                        torch.save(net.module.model.state_dict(), save_mode_path)
                    print(f"dice_avg_best:{dice_avg_best}, mIou_best:{miou_best}")


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
        label_list = range(1816)
        labeled_idxs = random.sample(label_list, args.label_num)
        unlabeled_idxs = [x for x in label_list if x not in labeled_idxs]
        test_num = 401
        r = range(test_num)
        test_idxs = random.sample(r, test_num)
        train_data_path = file_path + "devdata/new_dataset_2/annotation.json"  # todo

        dataset_lb = Throatdataset(args, train_data_path, index=labeled_idxs, aug_type='strong', mode="train",
                                   resize_shape=args.resize_shape)
        dataset_ulb = Throatdataset(args, train_data_path, index=unlabeled_idxs, aug_type="strong",
                                    mode="train",
                                    resize_shape=args.resize_shape)
        dataset_test = Throatdataset(args, train_data_path, index=test_idxs, aug_type="weak", mode="test",
                                     resize_shape=args.resize_shape)
    else:
        raise NotImplementedError

    mp.spawn(
        main,
        nprocs=args.gpus,
        args=(args, dataset_lb, dataset_ulb, dataset_test)
    )
