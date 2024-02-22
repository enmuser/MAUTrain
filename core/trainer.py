import csv
import os.path
import statistics

import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    ft = codecs.open(configs.gen_frm_dir + '/all_performance.txt', 'a+')
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
    ft.writelines('====================================start=====' + str(itr) + '=====start=============================================\n')
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
                    if shape[1] == 1:
                        # new_shape = (16,3,64,64)
                        new_shape = (shape[0], 3, *shape[2:])
                        # 将tensor按照某一维度扩展
                        t1.expand(new_shape)
                        t2.expand(new_shape)
                    elif shape[1] == 2:
                        # new_shape = (16,3,64,64)
                        new_shape = (shape[0], 1, *shape[2:])
                        add_channel = np.zeros(new_shape)
                        add_channel = torch.FloatTensor(add_channel).to(configs.device)
                        t1 = torch.concat([t1, add_channel], axis=1)
                        t2 = torch.concat([t2, add_channel], axis=1)
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
                ssim_list[i] = score
                avg_ssim += score
            # results/mau/performance.txt
            f.writelines('batch_id: '+str(batch_id) + '\n\n' +
                         'mse_list: \n' + str(mse_list) + '\n\n' 
                         'mae_list: \n'+str(mae_list) + '\n\n'+
                         'psnr_list: \n' + str(psnr_list) + '\n\n' +
                         'lpips_list: \n' + str(lpips_list) + '\n\n' +
                         'ssim_list: \n' + str(ssim_list) + '\n\n')
            f.writelines('============================================================================================\n')

            ft.writelines('batch_id: '+str(batch_id) + '\n\n' +
                         'mse_list: \n' + str(mse_list) + '\n' +' mse_list_avg: '+ str(statistics.mean(mse_list)) +'\n\n' 
                         'mae_list: \n'+str(mae_list) + '\n' + ' mae_list_avg: '+ str(statistics.mean(mae_list)) +'\n\n'+
                         'psnr_list: \n' + str(psnr_list) + '\n' +' psnr_list_avg: '+ str(statistics.mean(psnr_list)) +'\n\n' +
                         'lpips_list: \n' + str(lpips_list) +'\n' + ' lpips_list_avg: '+ str(statistics.mean(lpips_list)) +'\n\n' +
                         'ssim_list: \n' + str(ssim_list) + '\n' +' ssim_list_avg: '+  str(statistics.mean(ssim_list)) +'\n\n')
            ft.writelines('**************************************************************************************************\n')

            # res_width = 64
            res_width = configs.img_width
            # res_height = 64
            res_height = configs.img_height
            # img = (64 * 2 , 20 * 64, 1)
            interval = 4
            img = np.ones((2 * res_height,
                           configs.total_length * res_width,
                           configs.img_channel))
            img_input = np.ones((res_height,
                                 configs.input_length * res_width + configs.input_length * interval,
                                 configs.img_channel))
            img_ground_true = np.ones((res_height,
                                       configs.pred_length * res_width + configs.pred_length * interval,
                                       configs.img_channel))
            img_pred = np.ones((res_height,
                                configs.pred_length * res_width + configs.pred_length * interval,
                                configs.img_channel))
            if configs.is_training == True and configs.dataset == 'kth':
                img_input = np.ones((res_height,
                                     (configs.input_length//2) * res_width + (configs.input_length//2) * interval,
                                     configs.img_channel))

                img_ground_true = np.ones((res_height,
                                           (configs.pred_length//2) * res_width + (configs.pred_length//2) * interval,
                                           configs.img_channel))
                img_pred = np.ones((res_height,
                                    (configs.pred_length//2) * res_width +(configs.pred_length//2)  * interval,
                                    configs.img_channel))
            # name = 1.png
            name = str(batch_id) + '.png'

            img_input_name = str(batch_id) + '_input.png'
            img_ground_true_name = str(batch_id) + '_ground_true.png'
            img_pred_name = str(batch_id) + '_pred.png'

            # file_name = results/mau/1.png
            file_name = os.path.join(res_path, name)

            file_img_input_name = os.path.join(res_path, img_input_name)
            file_img_ground_true_name = os.path.join(res_path, img_ground_true_name)
            file_img_pred_name = os.path.join(res_path, img_pred_name)

            img_input_single = np.ones((res_height, res_width, configs.img_channel))
            img_ground_true_single = np.ones((res_height, res_width, configs.img_channel))
            img_pred_single = np.ones((res_height, res_width, configs.img_channel))

            # total_length = 20 | 0,1,2,3,...,17,18,19
            for i in range(configs.total_length):
                # img[:res_height, i * res_width:(i + 1) * res_width, :]
                # = img[:res_height, i * res_width:(i + 1) * res_width, :]
                # = img[:64,1*64:2*64,:] = test_ims[0, 1, :]
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :]

                if i < configs.input_length:
                    if configs.is_training == True and configs.dataset == 'kth':
                        if i % 2 != 0:
                            continue
                        else:
                            img_input[:res_height, ((i // 2) * res_width + (i // 2) * interval):(((i // 2) + 1) * res_width + (i // 2) * interval), :] = test_ims[0, i, :]
                    else:
                        img_input[:res_height, (i * res_width + i * interval):((i + 1) * res_width + i * interval),:] = test_ims[0, i, :]
                    if configs.img_channel == 2:
                        img_input_single[:, :, :] = test_ims[0, i, :]
                        img_total_input_single = img_input_single[:, :, 0] + img_input_single[:, :, 1]
                        img_input_name_single = 'batch_' + str(batch_id) + '_input_' + str(i) + '.svg'
                        file_img_input_name_single_svg = os.path.join(res_path, img_input_name_single)
                        plt.imsave(file_img_input_name_single_svg,img_total_input_single.reshape(img_total_input_single.shape[0],img_total_input_single.shape[1]), vmin=0, vmax=1.0)
                else:
                    if configs.is_training == True and configs.dataset == 'kth':
                        if (i - configs.input_length) % 2 != 0:
                            continue
                        else:
                            img_ground_true[:res_height, (((i - configs.input_length) // 2) * res_width + ((i - configs.input_length) // 2) * interval):((((i - configs.input_length) // 2) + 1) * res_width + ((i - configs.input_length) // 2) * interval),:] = test_ims[0, i, :]
                    else:
                        img_ground_true[:res_height,((i - configs.input_length) * res_width + (i - configs.input_length) * interval):((i + 1 - configs.input_length) * res_width + (i - configs.input_length) * interval),:] = test_ims[0, i, :]
                    if configs.img_channel == 2:
                        img_ground_true_single[:, :, :] = test_ims[0, i, :]
                        img_total_ground_true_single = img_ground_true_single[:, :, 0] + img_ground_true_single[:, :, 1]
                        img_ground_true_name_single = 'batch_' + str(batch_id) + '_ground_true_' + str(i - configs.input_length) + '.svg'
                        file_img_ground_true_name_single_svg = os.path.join(res_path, img_ground_true_name_single)
                        plt.imsave(file_img_ground_true_name_single_svg,img_total_ground_true_single.reshape(img_total_ground_true_single.shape[0],img_total_ground_true_single.shape[1]), vmin=0,vmax=1.0)

                        img_pred_single[:, :, :] = img_out[0, -output_length + (i - configs.input_length), :]
                        img_total_pred_single = img_pred_single[:, :, 0] + img_pred_single[:, :, 1]

                        img_total_target_pred_diff_single = img_total_ground_true_single[:, :] - img_total_pred_single[:, :]
                        img_target_pred_diff_name_single = 'batch_' + str(batch_id) + '_target_pred_diff_' + str(i - configs.input_length) + '.svg'
                        file_img_target_pred_diff_name_single_svg = os.path.join(res_path,img_target_pred_diff_name_single)
                        plt.imsave(file_img_target_pred_diff_name_single_svg,img_total_target_pred_diff_single.reshape(img_total_target_pred_diff_single.shape[0],img_total_target_pred_diff_single.shape[1]),vmin=0, vmax=1.0)
            # total_length = 10 | 0,1,2,3,...,7,8,9
            for i in range(output_length):
                # img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,:]
                # = img[64:, (10 + 1) * 64:(10 + 1 + 1) * 64,:] = img_out[0, -10 + 1, :] = img_out[0, -9, :]
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,:] \
                    = img_out[0, -output_length + i, :]
                if configs.is_training == True and configs.dataset == 'kth':
                    if i % 2 != 0:
                        continue
                    else:
                        img_pred[:res_height,((i // 2) * res_width + (i // 2) * interval):(((i // 2) + 1) * res_width + (i // 2) * interval),:] = img_out[0, -output_length + i, :]
                else:
                    img_pred[:res_height, (i * res_width + i * interval):((i + 1) * res_width + i * interval),:] = img_out[0, -output_length + i, :]
                if configs.img_channel == 2:
                    img_pred_single[:, :, :] = img_out[0, -output_length + i, :]
                    img_total_pred_single = img_pred_single[:, :, 0] + img_pred_single[:, :, 1]
                    img_pred_name_single = 'batch_' + str(batch_id) + '_pred_' + str(i) + '.svg'
                    file_img_pred_name_single_svg = os.path.join(res_path, img_pred_name_single)
                    plt.imsave(file_img_pred_name_single_svg, img_total_pred_single.reshape(img_total_pred_single.shape[0],img_total_pred_single.shape[1]),vmin=0, vmax=1.0)

            # 将小于0的变成0, 将大于1的变成1
            if configs.img_channel == 2:
                # add_image = np.zeros((2 * res_height,
                #          configs.total_length * res_width,1))
                # img = np.concatenate([img,add_image],axis=2)
                img_total = img[:, :, 0] + img[:, :, 1]
                name_svg = str(batch_id) + '.svg'

                img_total_input = img_input[:, :, 0] + img_input[:, :, 1]
                img_input_name = str(batch_id) + '_input.svg'

                img_total_ground_true = img_ground_true[:, :, 0] + img_ground_true[:, :, 1]
                img_ground_true_name = str(batch_id) + '_ground_true.svg'

                img_total_pred = img_pred[:, :, 0] + img_pred[:, :, 1]
                img_pred_name = str(batch_id) + '_pred.svg'

                # file_name = results/mau/1.png
                file_name_svg = os.path.join(res_path, name_svg)

                file_img_input_nam_svg = os.path.join(res_path, img_input_name)
                file_img_ground_true_name_svg = os.path.join(res_path, img_ground_true_name)
                file_img_pred_name_svg = os.path.join(res_path, img_pred_name)

                plt.imsave(file_name_svg, img_total.reshape(img_total.shape[0], img_total.shape[1]), vmin=0, vmax=1.0)

                plt.imsave(file_img_input_nam_svg, img_total_input.reshape(img_total_input.shape[0], img_total_input.shape[1]), vmin=0, vmax=1.0)
                plt.imsave(file_img_ground_true_name_svg, img_total_ground_true.reshape(img_total_ground_true.shape[0], img_total_ground_true.shape[1]), vmin=0, vmax=1.0)
                plt.imsave(file_img_pred_name_svg, img_total_pred.reshape(img_total_pred.shape[0], img_total_pred.shape[1]), vmin=0, vmax=1.0)
            else:
                img = np.maximum(img, 0)
                img = np.minimum(img, 1)

                img_input = np.maximum(img_input, 0)
                img_input = np.minimum(img_input, 1)

                img_ground_true = np.maximum(img_ground_true, 0)
                img_ground_true = np.minimum(img_ground_true, 1)

                img_pred = np.maximum(img_pred, 0)
                img_pred = np.minimum(img_pred, 1)

                # 写出对比图片
                cv2.imwrite(file_name, (img * 255).astype(np.uint8))
                #cv2.imwrite(file_img_input_name, (img_input * 255).astype(np.uint8))
                #cv2.imwrite(file_img_ground_true_name, (img_ground_true * 255).astype(np.uint8))
                #cv2.imwrite(file_img_pred_name, (img_pred * 255).astype(np.uint8))
            batch_id = batch_id + 1
    ft.writelines('====================================end=====' + str(itr) + '=====end=============================================\n')
    f.close()
    ft.close()
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

        with codecs.open(configs.gen_frm_dir + '/all_data.txt', 'a+') as all_data_write:
            all_data_write.writelines('------------------current itr : ' + str(itr) + '---------------------\n')
            all_data_write.writelines('total mse per frame: ' + str(avg_mse) + '\n')
            all_data_write.writelines('total mae per frame: ' + str(avg_mae) + '\n')
            all_data_write.writelines('total psnr per frame: ' + str(avg_psnr) + '\n')
            all_data_write.writelines('total ssim per frame: ' + str(avg_ssim) + '\n')
            all_data_write.writelines('total lpips per frame: ' + str(avg_lpips) + '\n')

        plot_generate(avg_lpips, avg_mae, avg_mse, avg_psnr, avg_ssim, configs, itr)



def plot_generate(avg_lpips, avg_mae, avg_mse, avg_psnr, avg_ssim, configs, itr):
    data_list = []
    data_list.append(itr)
    data_list.append(avg_mse)
    data_list.append(avg_mae)
    data_list.append(avg_psnr)
    data_list.append(avg_ssim)
    data_list.append(avg_lpips)
    with open(configs.gen_frm_dir + '/result.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_list)
    result_file = open(configs.gen_frm_dir + '/result.csv')  # 打开csv文件
    result_reader = csv.reader(result_file)  # 读取csv文件
    result_data = list(result_reader)  # csv数据转换为列表
    length_row = len(result_data)  # 得到数据行数
    length_col = len(result_data[0])  # 得到每行长度
    itrList = list()
    mseList = list()
    maeList = list()
    psnrList = list()
    ssimList = list()
    lpipsList = list()
    for i in range(0, length_row):  # 从第二行开始读取
        itrList.append(int(result_data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
        mseList.append(float("{:.3f}".format(float(result_data[i][1]))))
        maeList.append(float("{:.3f}".format(float(result_data[i][2]))))
        psnrList.append(float("{:.3f}".format(float(result_data[i][3]))))
        ssimList.append(float("{:.3f}".format(float(result_data[i][4]))))
        lpipsList.append(float("{:.3f}".format(float(result_data[i][5]))))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.subplot(5, 1, 1)
    plt.plot(itrList, mseList, color='deepskyblue')
    plt.title("mse")
    plt.subplot(5, 1, 2)
    plt.plot(itrList, maeList, color='orange')
    plt.title("mae")
    plt.subplot(5, 1, 3)
    plt.plot(itrList, psnrList, color='green')
    plt.title("psnr")
    plt.subplot(5, 1, 4)
    plt.plot(itrList, ssimList, color='red')
    plt.title("ssim")
    plt.subplot(5, 1, 5)
    plt.plot(itrList, lpipsList, color='aquamarine')
    plt.title("lpips")
    plt.savefig(configs.gen_frm_dir + 'plot/all_result_plot_' + str(itr) + '.png')
    try:
        os.remove(configs.gen_frm_dir + 'plot/all_result_plot_' + str(itr - int(configs.test_interval)) + '.png')
    except:
        print("file not found!")
    plt.close()
    plt.plot(itrList, mseList, color='deepskyblue')
    plt.title("mse")
    for a, b in zip(itrList, mseList):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    plt.savefig(configs.gen_frm_dir + 'plot/mse_result_plot_' + str(itr) + '.png')
    try:
        os.remove(configs.gen_frm_dir + 'plot/mse_result_plot_' + str(itr - int(configs.test_interval)) + '.png')
    except:
        print("file not found!")
    plt.close()
    plt.plot(itrList, maeList, color='orange')
    plt.title("mae")
    for a, b in zip(itrList, maeList):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    plt.savefig(configs.gen_frm_dir + 'plot/mae_result_plot_' + str(itr) + '.png')
    try:
        os.remove(configs.gen_frm_dir + 'plot/mae_result_plot_' + str(itr - int(configs.test_interval)) + '.png')
    except:
        print("file not found!")
    plt.close()
    plt.plot(itrList, psnrList, color='green')
    plt.title("psnr")
    for a, b in zip(itrList, psnrList):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    plt.savefig(configs.gen_frm_dir + 'plot/psnr_result_plot_' + str(itr) + '.png')
    try:
        os.remove(
            configs.gen_frm_dir + 'plot/psnr_result_plot_' + str(itr - int(configs.test_interval)) + '.png')
    except:
        print("file not found!")
    plt.close()
    plt.plot(itrList, ssimList, color='red')
    plt.title("ssim")
    for a, b in zip(itrList, ssimList):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    plt.savefig(configs.gen_frm_dir + 'plot/ssim_result_plot_' + str(itr) + '.png')
    try:
        os.remove(
            configs.gen_frm_dir + 'plot/ssim_result_plot_' + str(itr - int(configs.test_interval)) + '.png')
    except:
        print("file not found!")
    plt.close()
    plt.plot(itrList, lpipsList, color='aquamarine')
    plt.title("lpips")
    for a, b in zip(itrList, lpipsList):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    plt.savefig(configs.gen_frm_dir + 'plot/lpips_result_plot_' + str(itr) + '.png')
    try:
        os.remove(
            configs.gen_frm_dir + 'plot/lpips_result_plot_' + str(itr - int(configs.test_interval)) + '.png')
    except:
        print("file not found!")
    plt.close()
