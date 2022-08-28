import os.path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import torch
import codecs
import lpips


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    # display_interval = 1 打印损失的频次
    if itr % configs.display_interval == 0:
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))


def test(model, test_input_handle, configs, itr):
    print('test...')
    loss_fn = lpips.LPIPS(net='alex', spatial=True).to(configs.device)
    # gen_frm_dir = results/mau/
    res_path = configs.gen_frm_dir + '/' + str(itr)

    if not os.path.exists(res_path):
        os.mkdir(res_path)
    f = codecs.open(res_path + '/performance.txt', 'w+')
    f.truncate()

    avg_mse = 0
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    batch_id = 0
    img_mse, img_mae, img_psnr, ssim, img_lpips, mse_list, mae_list, psnr_list, ssim_list, lpips_list = [], [], [], [], [], [], [], [], [], []
    # total_length = 20 , input_length = 10
    # total_length - input_length = 10 , 0,1,2,...,8,9
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        img_mae.append(0)
        img_psnr.append(0)
        ssim.append(0)
        img_lpips.append(0)

        mse_list.append(0)
        mae_list.append(0)
        psnr_list.append(0)
        ssim_list.append(0)
        lpips_list.append(0)
    # max_epoches = 200000
    for epoch in range(configs.max_epoches):
        # num_save_samples = 5
        # batch_id 可以为 0,1,2,3,4,5 当 batch_id = 6时终止，
        if batch_id > configs.num_save_samples:
            break
        # num_save_samples = 5
        for data in test_input_handle:
            if batch_id > configs.num_save_samples:
                break
            print(batch_id)

            batch_size = data.shape[0]
            # real_input_flag = 16 * 19 * 64 * 64 * 1
            real_input_flag = np.zeros(
                (batch_size,
                 configs.total_length - configs.input_length - 1,
                 configs.img_height // configs.patch_size,
                 configs.img_width // configs.patch_size,
                 configs.patch_size ** 2 * configs.img_channel))
            # data = 16 * 20 * 1 * 64 * 64
            # img_gen = # 16 * 19 * 1 * 64 * 64
            img_gen = model.test(data, real_input_flag)
            # img_gen = 16 * 19 * 1 * 64 * 64 -> 16 * 19 * 64 * 64 * 1
            img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            # data = 16 * 20 * 1 * 64 * 64 -> 16 * 20 * 64 * 64 * 1 = test_ims
            # test_ims =  16 * 20 * 64 * 64 * 1
            test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            # output_length = total_length(20) - input_length(10) = 10
            output_length = configs.total_length - configs.input_length
            # output_length = min(10,19) = 10
            output_length = min(output_length, configs.total_length - 1)
            # test_ims = 16 * 20 * 64 * 64 * 1, patch_size = 1
            # 输出是:  test_ims = 16 * 20 * 64 * 64 * 1
            test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
            # 输出是:  img_gen = 16 * 19 * 64 * 64 * 1
            img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
            # img_out = img_gen[:,-10,:] = 16 * 10 * 64 * 64 * 1
            img_out = img_gen[:, -output_length:, :]

            # MSE per frame
            # output_length = 10 | 0, 1, 2, 3,..., 8, 9
            for i in range(output_length):
                # x = test_ims[:, 0 + 10, :] = 16 * 64 * 64 * 1
                # x = test_ims[:, 1 + 10, :] = 16 * 64 * 64 * 1
                # x = test_ims[:, 2 + 10, :] = 16 * 64 * 64 * 1
                # ......
                # 一次取后十帧 与 预测的帧进行比较
                x = test_ims[:, i + configs.input_length, :]
                # 取对应的预测帧
                # gx = 16 * 64 * 64 * 1
                gx = img_out[:, i, :]
                # np.maximum 逐个对比选择较大的哪个
                gx = np.maximum(gx, 0)
                # np.maximum 逐个对比选择较小的哪个
                # 将小于0的变成0, 将大于1的变成1
                gx = np.minimum(gx, 1)
                # 均方误差
                mse = np.square(x - gx).sum()/batch_size
                # 平均绝对误差
                mae = np.abs(x - gx).sum()/batch_size
                psnr = 0
                # 学习感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS)
                t1 = torch.from_numpy((x - 0.5) / 0.5).to(configs.device)
                # t1 = 16 * 64 * 64 * 1 -> 16 * 1 * 64 * 64
                t1 = t1.permute((0, 3, 1, 2))
                t2 = torch.from_numpy((gx - 0.5) / 0.5).to(configs.device)
                # t2 = 16 * 64 * 64 * 1 -> 16 * 1 * 64 * 64
                t2 = t2.permute((0, 3, 1, 2))
                # shape =  16 * 1 * 64 * 64
                shape = t1.shape
                # shape[1] = 1
                if not shape[1] == 3:
                    # new_shape = (16,3,64,64)
                    new_shape = (shape[0], 3, *shape[2:])
                    # 将tensor按照某一维度扩展
                    t1.expand(new_shape)
                    t2.expand(new_shape)
                d = loss_fn.forward(t1, t2)
                lpips_score = d.mean()
                lpips_score = lpips_score.detach().cpu().numpy() * 100
                # 峰值信噪比(Peak Signal to Noise Ratio, PSNR)
                # batch_size = 16 | 0,1,2,3,...,13,14,15
                for sample_id in range(batch_size):
                    # 计算每个批次的 mse
                    mse_tmp = np.square(
                        x[sample_id, :] - gx[sample_id, :]).mean()
                    # 累加 mse
                    psnr += 10 * np.log10(1 / mse_tmp)
                # 除以批次大小 获得平均 psnr
                psnr /= (batch_size)
                # ------------------------------
                img_mse[i] += mse
                img_mae[i] += mae
                img_psnr[i] += psnr
                img_lpips[i] += lpips_score
                # ------------------------------
                mse_list[i] = mse
                mae_list[i] = mae
                psnr_list[i] = psnr
                lpips_list[i] = lpips_score
                # ------------------------------
                avg_mse += mse
                avg_mae += mae
                avg_psnr += psnr
                avg_lpips += lpips_score
                # ssim 结构相似性指数（structural similarity index，SSIM）
                score = 0
                # batch_size = 16 | 0,1,2,3,...,13,14,15
                for b in range(batch_size):
                    # 计算每一个批次的ssim, 并且累加
                    score += compare_ssim(x[b, :], gx[b, :], multichannel=True)
                # 除以批次大小 获得平均ssim
                score /= batch_size
                # ------------------------------
                ssim[i] += score
                ssim_list[i] += score
                avg_ssim += score
            # results/mau/performance.txt
            f.writelines('batch_id: '+str(batch_id) + '\n\n' +
                         'mse_list: \n' + str(mse_list) + '\n\n' 
                         'mae_list: \n'+str(mae_list) + '\n\n'+
                         'psnr_list: \n' + str(psnr_list) + '\n\n' +
                         'lpips_list: \n' + str(lpips_list) + '\n\n' +
                         'ssim_list: \n' + str(ssim_list) + '\n\n')
            f.writelines('============================================================================================\n')
            # res_width = 64
            res_width = configs.img_width
            # res_height = 64
            res_height = configs.img_height
            # img = (64 * 2 , 20 * 64, 1)
            img = np.ones((2 * res_height,
                           configs.total_length * res_width,
                           configs.img_channel))
            # name = 1.png
            name = str(batch_id) + '.png'
            # file_name = results/mau/1.png
            file_name = os.path.join(res_path, name)
            # total_length = 20 | 0,1,2,3,...,17,18,19
            for i in range(configs.total_length):
                # img[:res_height, i * res_width:(i + 1) * res_width, :]
                # = img[:res_height, i * res_width:(i + 1) * res_width, :]
                # = img[:64,1*64:2*64,:] = test_ims[0, 1, :]
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :]
            # total_length = 10 | 0,1,2,3,...,7,8,9
            for i in range(output_length):
                # img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,:]
                # = img[64:, (10 + 1) * 64:(10 + 1 + 1) * 64,:] = img_out[0, -10 + 1, :] = img_out[0, -9, :]
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,:] \
                    = img_out[0, -output_length + i, :]
            # 将小于0的变成0, 将大于1的变成1
            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            # 写出对比图片
            cv2.imwrite(file_name, (img * 255).astype(np.uint8))
            batch_id = batch_id + 1
    f.close()
    # results/mau/data.txt
    with codecs.open(res_path + '/data.txt', 'w+') as data_write:
        data_write.truncate()
        # 求得每一帧的 平均mse
        # avg_mse = avg_mse / (6 * 10)
        avg_mse = avg_mse / (batch_id * output_length)
        print('mse per frame: ' + str(avg_mse))
        # total_length - input_length = 20 - 10 = 10 | 0,1,2,3,...,7,8,9
        # 因为 img_mse[i] += mse, 每一个img_mse[i]累加了6个批次对应帧位的mse
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / batch_id)
            # 求得6个批次下之后, 10个帧位,每一帧的平均mse
            img_mse[i] = img_mse[i] / batch_id
        data_write.writelines('total mse per frame: ' + str(avg_mse) + '\n\n')
        data_write.writelines('10 location mse per frame: \n' + str(img_mse) + '\n')
        data_write.writelines('-----------------------------------------------------------------\n')

        # 求得每一帧的 平均mae
        #  avg_mae = avg_mae / (6 * 10)
        avg_mae = avg_mae / (batch_id * output_length)
        print('mae per frame: ' + str(avg_mae))
        # total_length - input_length = 20 - 10 = 10 | 0,1,2,3,...,7,8,9
        # 因为 img_mae[i] += mae, 每一个img_mae[i]累加了6个批次对应帧位的mae
        for i in range(configs.total_length - configs.input_length):
            print(img_mae[i] / batch_id)
            # 求得6个批次下之后, 10个帧位,每一帧的平均mae
            img_mae[i] = img_mae[i] / batch_id
        data_write.writelines('total mae per frame: ' +str(avg_mae) + '\n\n')
        data_write.writelines('10 location mae per frame: \n' + str(img_mae) + '\n')
        data_write.writelines('-----------------------------------------------------------------\n')
        # 求得每一帧的 平均psnr
        #  avg_psnr = avg_psnr / (6 * 10)
        avg_psnr = avg_psnr / (batch_id * output_length)
        print('psnr per frame: ' + str(avg_psnr))
        # total_length - input_length = 20 - 10 = 10 | 0,1,2,3,...,7,8,9
        # 因为 img_psnr[i] += psnr, 每一个img_psnr[i]累加了6个批次对应帧位的 psnr
        for i in range(configs.total_length - configs.input_length):
            print(img_psnr[i] / batch_id)
            # 求得6个批次下之后, 10个帧位,每一帧的平均psnr
            img_psnr[i] = img_psnr[i] / batch_id
        data_write.writelines('total psnr per frame: ' +str(avg_psnr) + '\n\n')
        data_write.writelines('10 location psnr per frame: \n' + str(img_psnr) + '\n')
        data_write.writelines('-----------------------------------------------------------------\n')
        # 求得每一帧的 平均ssim
        # avg_ssim = avg_ssim / (6 * 10)
        avg_ssim = avg_ssim / (batch_id * output_length)
        print('ssim per frame: ' + str(avg_ssim))
        # total_length - input_length = 20 - 10 = 10 | 0,1,2,3,...,7,8,9
        # 因为 ssim[i] += score, 每一个ssim[i]累加了6个批次对应帧位的 ssim
        for i in range(configs.total_length - configs.input_length):
            print(ssim[i] / batch_id)
            # 求得6个批次下之后, 10个帧位,每一帧的平均ssim
            ssim[i] = ssim[i] / batch_id
        data_write.writelines('total ssim per frame: ' +str(avg_ssim) + '\n\n')
        data_write.writelines('10 location ssim per frame: \n' + str(ssim) + '\n')
        data_write.writelines('-----------------------------------------------------------------\n')
        # 求得每一帧的 平均lpips
        # avg_lpips = avg_lpips / (6 * 10)
        avg_lpips = avg_lpips / (batch_id * output_length)
        print('lpips per frame: ' + str(avg_lpips))
        # total_length - input_length = 20 - 10 = 10 | 0,1,2,3,...,7,8,9
        # 因为 img_lpips[i] += lpips_score, 每一个img_lpips[i]累加了6个批次对应帧位的lpips
        for i in range(configs.total_length - configs.input_length):
            print(img_lpips[i] / batch_id)
            # 求得6个批次下之后, 10个帧位,每一帧的平均lpips
            img_lpips[i] = img_lpips[i] / batch_id
        data_write.writelines('total lpips per frame: ' +str(avg_lpips) + '\n\n')
        data_write.writelines('10 location lpips per frame: \n' + str(img_lpips) + '\n')
