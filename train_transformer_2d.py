## -*- coding: utf-8 -*-
import argparse
import datetime
import importlib
import logging
import shutil
import sys
from pathlib import Path
import utils.LoaderFish as LoaderFish

from tqdm import tqdm

from utils.util import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='transformer_2d_tps', help='model name [default: transformer]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8]')
    parser.add_argument('--epoch', default=20, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.002]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=2e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=1024, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int, default=3, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--deform_level', type=float, default=0.4, help='tps deform level')
    return parser.parse_args()


def main(args):
    logger = logging.getLogger("Transformer")

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('transformer')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    visual_dir = experiment_dir.joinpath("visual")
    visual_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('INIT PARAMETERS ...')
    log_string(args)

    TRAIN_DATASET=LoaderFish.PointRegDataset(total_data=100000,deform_level=args.deform_level)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, drop_last=True)
    TEST_DATASET=LoaderFish.PointRegDataset(total_data=3000,deform_level=args.deform_level)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                                 drop_last=True)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('utils/util.py', str(experiment_dir))
    channel = 3
    classifier = MODEL.get_model(d_model=128, channel=channel).cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        epoch = checkpoint['epoch']
        model_loss = checkpoint['loss']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        epoch = 0
        model_loss = 999.0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size


    ###   仔细检查训练流程
    for epoch in range(epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        #  Train one epoch
        total_loss1, total_loss2, i = 0, 0, 0
        for data in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            i += 1

            points1, points2, _,_ = data
            # print(points1.size())   #[B,2,N]
            shape=list(points1.size())
            shape[1]=1
            zero=torch.zeros(shape,dtype=points1.dtype)
            points1,points2=torch.cat([points1,zero],dim=1),torch.cat([points2,zero],dim=1)
            points1,points2=points1.permute(0,2,1),points2.permute(0,2,1)

            points1, points2 = points1.cuda(), points2.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            ret = classifier(points1, points2,epoch=epoch,total_epoch=args.epoch)
            if len(ret) == 2:
                warped, loss1 = ret
                loss2=torch.zeros([1])
            elif len(ret) == 3:
                warped, loss1, loss2 = ret
            if epoch < 10:
                # loss = (loss1 + loss2).mean()
                loss = loss1.mean()
            else:
                # loss = (loss1 + 0.1*loss2).mean()
                loss = loss1.mean()
            total_loss1 += float(loss1.mean())
            total_loss2 += float(loss2.mean())
            loss.backward()
            optimizer.step()

        log_string('EPOCH %d train: loss1 is: %.5f   loss2 is: %.5f ' % (epoch + 1, total_loss1 / i, total_loss2 / i))
        with torch.no_grad():
            total_loss1, total_loss2, i = 0, 0, 0
            for batch_id, (points1, points2, _,_) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                          smoothing=0.9):
                i += 1

                # print(points1.size())   #[B,2,N]
                shape = list(points1.size())
                shape[1] = 1
                zero = torch.zeros(shape, dtype=points1.dtype)
                points1, points2 = torch.cat([points1, zero], dim=1), torch.cat([points2, zero], dim=1)
                points1, points2 = points1.permute(0, 2, 1), points2.permute(0, 2, 1)

                points1, points2 = points1.cuda(), points2.cuda()
                classifier = classifier.eval()
                ret = classifier(points1, points2, epoch=epoch, total_epoch=args.epoch)
                if len(ret) == 2:
                    warped, loss1 = ret
                    loss2=torch.zeros([1])
                elif len(ret) == 3:
                    warped, loss1, loss2 = ret
                total_loss1 += float(loss1.mean())
                total_loss2 += float(loss2.mean())

            log_string(
                'EPOCH %d test: loss1 is: %.5f   loss2 is: %.5f ' % (epoch + 1, total_loss1 / i, total_loss2 / i))

            if (total_loss1 / batch_id) < model_loss:
                model_loss = total_loss1 / batch_id
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'loss': model_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

    #   save the last epoch
    model_loss = total_loss1 / batch_id
    logger.info('Save last model...')
    savepath = str(checkpoints_dir) + '/last_model.pth'
    log_string('Saving at %s' % savepath)
    state = {
        'epoch': epoch,
        'loss': model_loss,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


if __name__ == '__main__':
    args = parse_args()
    main(args)
