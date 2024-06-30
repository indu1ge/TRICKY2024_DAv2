import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.Booster import Booster
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss, SILogMSELoss
from util.metric import eval_depth
from util.utils import init_log
import cv2
from util.alignment import align_depth_least_square_torch
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti', 'booster'])
parser.add_argument('--img_size', default=518, type=int)
parser.add_argument('--min_depth', default=0.001, type=float)
parser.add_argument('--max_depth', default=10000, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained_from', type=str)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'booster':
        trainset = Booster('dataset/splits/booster/booster_train_mask1.txt', 'train', size=size)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    # if args.dataset == 'hypersim':
    #     valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    # elif args.dataset == 'vkitti':
    #     valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    # else:
    #     raise NotImplementedError
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        # model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu'), strict=False)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    # criterion = SiLogLoss().cuda(local_rank)
    criterion = torch.nn.MSELoss()
    # criterion = SILogMSELoss(lamb=0.5).cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    # previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        # if rank == 0:
        #     logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
        #     logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
        #                 'log10: {:.3f}, silog: {:.3f}'.format(
        #                     epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
        #                     previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(tqdm(trainloader,desc=f'train')):
            optimizer.zero_grad()
            
            img, depth, valid_mask, tom_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['mask'].cuda(), sample['semseg_mask'].cuda()
            
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
                tom_mask = tom_mask.flip(-1)
            
            pred = model(img)
            # path = sample['image_path']
            # img1 = cv2.imread(path[0])
            # pred_infer = noddpmodel.infer_image(img1)

            scale, shift = align_depth_least_square_torch(
                gt_arr=depth.detach(),
                pred_arr=pred.detach(),
                valid_mask_arr=(valid_mask & (depth >= 1e-3)).detach(),
                return_scale_shift=True,
            )

            pred = pred * scale + shift
            
            valid_mask1 = valid_mask & (depth >= 1e-3) & (pred >= 1e-3)
            loss = criterion(pred[valid_mask1], depth[valid_mask1])
            # mask1 = tom_mask & valid_mask1
            # loss += 0.1 * criterion(pred[mask1], depth[mask1])
            # loss += 0.1 * criterion(pred, depth, tom_mask & (depth >= 1e-3) & (pred >= 1e-3))
            # mask1 = tom_mask & (depth >= 1e-3)
            # loss = criterion(pred[mask1], depth[mask1])

            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 10 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        # model.eval()
        
        # results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
        #            'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
        #            'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        # nsamples = torch.tensor([0.0]).cuda()
        
        # for i, sample in enumerate(valloader):
            
        #     img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            
        #     with torch.no_grad():
        #         pred = model(img)
        #         pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
        #     valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
        #     if valid_mask.sum() < 10:
        #         continue
            
        #     cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
        #     for k in results.keys():
        #         results[k] += cur_results[k]
        #     nsamples += 1
        
        # torch.distributed.barrier()
        
        # for k in results.keys():
        #     dist.reduce(results[k], dst=0)
        # dist.reduce(nsamples, dst=0)
        
        # if rank == 0:
        #     logger.info('==========================================================================================')
        #     logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        #     logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
        #     logger.info('==========================================================================================')
        #     print()
            
        #     for name, metric in results.items():
        #         writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        # for k in results.keys():
        #     if k in ['d1', 'd2', 'd3']:
        #         previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        #     else:
        #         previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if epoch > 20 and (epoch + 1) % 5==0:
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    # 'previous_best': previous_best,
                }
                ckpt_save_path = f'epoch_{epoch:02d}'
                ckpt_save_path = os.path.join(args.save_path, ckpt_save_path)
                torch.save(checkpoint, ckpt_save_path)
                # torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()