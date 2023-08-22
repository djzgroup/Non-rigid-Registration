## -*- coding: utf-8 -*-
import argparse
import datetime
import importlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from utils.TPS3d_dataset import TPS3d_dataset
from utils.cloth_dataset import cloth_dataset
from utils.util import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='transformer_3d_tps', help='model name [default: transformer]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--epoch', default=1, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--npoint', type=int, default=512, help='Point Number [default: 2048]')
    parser.add_argument('--drop_num', type=int, default=None, )
    parser.add_argument('--deform_level', type=float, default=0.3, )
    parser.add_argument('--dataset', type=str, default="tps", choices=["tps", "cloth"])
    parser.add_argument('--out_liner_num', type=int, default=None, )
    parser.add_argument('--noise',default=False,action='store_true')
    parser.add_argument('--unoise',default=False,action='store_true')
    return parser.parse_args()


def main(args):
    logger = logging.getLogger("Transformer")

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('../log/')
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

    # TRAIN_DATASET = TPS3d_dataset(point_size=args.npoint, total_data=100, deform_level=0.3)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
    #                                               num_workers=0, drop_last=True)
    if args.dataset == "tps":
        TEST_DATASET = TPS3d_dataset(point_size=args.npoint, total_data=100, deform_level=args.deform_level,
                                     drop_num=args.drop_num,out_liner_num= args.out_liner_num,noise=args.noise,unoise=args.unoise)
    elif args.dataset == "cloth":
        TEST_DATASET = cloth_dataset(args.npoint, 100, False, False)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                                 drop_last=True)
    # log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    MODEL = importlib.import_module(args.model)
    # shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('utils.py', str(experiment_dir))

    classifier = MODEL.get_model(d_model=128, channel=3, npoint=args.npoint).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    print("loading checkpoint...")
    epoch = checkpoint['epoch']
    model_loss = checkpoint['loss']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string('Use pretrain model')

    t = ThreadPoolExecutor(args.batch_size * 5)

    ###   仔细检查训练流程
    for epoch in range(epoch, epoch + 1):

        visual_list = []
        classifier = classifier.eval()
        with torch.no_grad():
            total_loss1, total_loss2, i = 0, 0, 0
            counter = 0
            for batch_id, (points1, points2, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                        smoothing=0.9):
                counter += 1
                if counter >= 100:
                    break
                i += 1
                points1, points2 = points1.cuda(), points2.cuda()
                # warped, loss1 ,_= classifier(points1, points2)
                ret=classifier(points1,points2)
                if len(ret)==2:
                    warped,loss1=ret
                if len(ret)==3:
                    warped,loss1,_=ret
                if warped.size()[-1] != 3:
                    warped = warped.permute(0, 2, 1)
                total_loss1 += float(loss1.mean())
                visual_list.append([points1, points2, warped])

            for b in visual_list:
                points1, points2, warped = b
                t.submit(save_pc2visual, visual_dir, epoch, points1, points2, warped)
        del visual_list


if __name__ == '__main__':
    args = parse_args()
    main(args)
