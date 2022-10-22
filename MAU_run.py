import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
import pynvml


pynvml.nvmlInit()
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MAU')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--is_train', type=str, default='True', required=False)
args_main = parser.parse_args()
args_main.tied = True

if args_main.is_train == 'True':
    from configs.mnist_train_configs import configs
else:
    from configs.mnist_configs import configs

parser = configs()
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
args.tied = True




def schedule_sampling(eta, itr, channel, batch_size):
    #zero= 16 * 9 * 64 * 64 * 1
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    # scheduled_sampling = True
    # 是否进行计划采样
    if not args.scheduled_sampling:
        return 0.0, zeros
    # sampling_stop_iter = 50000
    # sampling_changing_rate = 0.00002
    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    print('eta: ', eta)
    # random_flip = 16 * 9
    # total_length = 20
    # input_length = 10
    # 生成 [0.0, 1.0) 随机数
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    # true_token = 16 * 9
    true_token = (random_flip < eta)
    # ones or zero = 64 * 64 * 1
    # img_height = 64
    # img_width = 64
    # patch_size = 1
    # channel = 1
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    real_input_flag = []
    # batch_size = 16
    # args.total_length - args.input_length - 1 = 9
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * channel))
    # real_input_flag =  16 * 9 * 64 * 64 * 1
    return eta, real_input_flag


def train_wrapper(model):
    begin = 0
    #实时显示显存大小
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    #是否存在预训练模型，如果存在加载预训练模型
    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)
    # sampling_start_value = 1.0
    eta = args.sampling_start_value
    #sampling_changing_rate =  0.00002
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    # real_input_flag = {}
    # max_epoches = 200000
    for epoch in range(0, args.max_epoches):
        # max_iterations = 80000
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:
            if itr > args.max_iterations:
                break
            batch_size = ims.shape[0]
            # real_input_flag = 16 * 9 * 64 * 64 * 1
            # 计划采样 schedule_sampling
            eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size)
            # test_interval = 1000 每1000次测试一次
            if itr % args.test_interval == 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)
            trainer.train(model, ims, real_input_flag, args, itr)
            # snapshot_interval = 1000 每1000次保存一次
            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)
            itr += 1
            #打印内存信息
            meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)


if __name__ == '__main__':

    print('Initializing models')
    #判断是训练还是测试
    if args.is_training == 'True':
        args.is_training = True
    else:
        args.is_training = False

    model = Model(args)

    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        test_wrapper(model)
