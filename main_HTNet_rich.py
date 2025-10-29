from os import path
import os
import numpy as np
import cv2
import time
import threading
from datetime import datetime, timedelta

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch
from Model import HTNet
import numpy as np
from facenet_pytorch import MTCNN
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

import copy

# Some of the codes are adapted from STSNet
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

#计算混淆矩阵
def confusionMatrix(gt, pred, show=False):  
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

#计算UF1/UAR
def recognition_evaluation(final_gt, final_pred, show=False):    
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''

# 获取人脸图片特征点
def whole_face_block_coordinates():
    df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
    m, n = df.shape #行数， 列数
    base_data_src = './datasets/combined_datasets_whole'
    total_emotion = 0
    image_size_u_v = 28
    # get the block center coordinates
    face_block_coordinates = {}

    # 遍历所有图
    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(
            df['filename_o'][i]) + ' .png'
        # print(image_name)
        img_path_apex = base_data_src + '/' + df['imagename'][i]
        train_face_image_apex = cv2.imread(img_path_apex) # (444, 533, 3)
        # 缩小图像至28-28
        face_apex = cv2.resize(train_face_image_apex, (28,28), interpolation=cv2.INTER_AREA)
        # 初始化人脸检测器MTCNN
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        # if not detecting face
        if batch_landmarks is None:
            batch_landmarks = np.array(
                [
                    [
                        [9.528073, 11.062551], 
                        [21.396168, 10.919773], 
                        [15.380184, 17.380562], 
                        [10.255435, 22.121233], 
                        [20.583706, 22.25584]
                    ]
                ]
            )

        # batch_landmarks 是一个1,1,5,2形状的数组， 1,1 代表第几张图的第几个人脸，5代表关键点个数，2代表坐标  选第0个元素对应的就是第一张图
        row_n, col_n = np.shape(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        # print(batch_landmarks[0])
        # get the block center coordinates
        face_block_coordinates[image_name] = batch_landmarks[0]
    # print(len(face_block_coordinates))
    return face_block_coordinates

# 沿特征点对图像进行切割(存在部分重叠)
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()
    # print(len(face_block_coordinates_dict))
    # Get train dataset
    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}
    # print(whole_optical_flow_imgs[0]) #spNO.189_f_150.png
    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        # nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
        #         four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        nose = ''
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
        # print(np.shape(l_eye))
    # print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs

def main(config):
    learning_rate = 0.00005
    batch_size = 256
    epochs = 800
    # epochs = 1600
    all_accuracy_dict = {}

    path_name = "new_test"

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        # device = torch.device('cpu')
        raise RuntimeError("CUDA不可用，程序需要GPU才能运行")

    # 交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    if (config.train):
        if not path.exists(path_name):
            os.mkdir(path_name)

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    #统计数据初始化
    total_gt = []
    total_pred = []
    best_total_pred = []
    t = time.time()

    main_path = './datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    # subName = ["031"]
    all_five_parts_optical_flow = crop_optical_flow_block()
    print(subName)

    # 初始化Rich控制台和进度条
    console = Console()
    # 训练进度跟踪变量
    total_subjects = len(subName)
    
    #Sub内循环
    for subject_idx, n_subName in enumerate(subName):
        print('Subject:%s(%s)' % (n_subName, 'Train' if config.train else 'Test'))

        # 数据初始化
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []       
        
        # 数据处理(感觉可以优化？)
        # 读取训练集
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))
                # l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                # r_eye_lips  =  cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                # lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                l_eye_lips = cv2.vconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips  =  cv2.vconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.hconcat([l_eye_lips, r_eye_lips])
                four_parts_train.append(lr_eye_lips)


        # 读取测试集
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                # l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                # r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                # lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                l_eye_lips = cv2.vconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.vconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.hconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)

        # 模型读取
        model = HTNet(
            image_size=28,
            patch_size=7,
            dim=256,  # 256,--96, 56-, 192
            heads=3,  # 3 ---- , 6-
            num_hierarchies=3,  # 3----number of hierarchies
            block_repeats=(2, 2, 10),#(2, 2, 8),------
            # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
            num_classes=3
        )
        model = model.to(device)

        weight_path = path_name + '/' + n_subName + '.pth'
        # Test时读取参数
        if not (config.train):
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        four_parts_train =  torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size)
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)
        # store best results
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        # 创建进度条和实时显示
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=1
        ) as progress:
            
            # 添加总体进度任务
            total_task = progress.add_task(
                f"[cyan]总体进度 ({subject_idx + 1}/{total_subjects})", 
                total=total_subjects * epochs
            )
            
            # 添加当前受试者进度任务
            subject_task = progress.add_task(
                f"[green]受试者 {n_subName}", 
                total=epochs
            )
            
            # 定时更新变量
            last_update_time = time.time()
            update_interval = 30  # 30秒更新一次
            epoch_times = []  # 存储每个epoch的时间
            
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                
                if (config.train):
                    # Training
                    model.train()
                    train_loss = 0.0
                    num_train_correct = 0
                    num_train_examples = 0

                    for batch_idx, batch in enumerate(train_dl):
                        optimizer.zero_grad()
                        x = batch[0].to(device)
                        y = batch[1].to(device)
                        yhat = model(x)
                        loss = loss_fn(yhat, y)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.data.item() * x.size(0)
                        num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                        num_train_examples += x.shape[0]

                    train_acc = num_train_correct / num_train_examples
                    train_loss = train_loss / len(train_dl.dataset)

                # Testing
                model.eval()
                val_loss = 0.0
                num_val_correct = 0
                num_val_examples = 0
                for batch in test_dl:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    val_loss += loss.data.item() * x.size(0)
                    num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_val_examples += y.shape[0]

                val_acc = num_val_correct / num_val_examples
                val_loss = val_loss / len(test_dl.dataset)
                
                # 记录当前epoch时间
                epoch_end_time = time.time()
                current_epoch_time = epoch_end_time - epoch_start_time
                epoch_times.append(current_epoch_time)
                
                # 计算平均每epoch时间
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                
                # 计算预计完成时间
                current_time = time.time()              
                
                # 更新进度条
                progress.update(subject_task, completed=epoch)
                progress.update(total_task, completed=(subject_idx * epochs) + epoch)
                
                # 每30秒或每个epoch结束时更新描述信息
                if current_time - last_update_time >= update_interval or epoch == epochs:
                    progress.update(
                        subject_task, 
                        description=f"[green]受试者 {n_subName} | Epoch {epoch}/{epochs} | 平均每Epoch: {avg_epoch_time:.2f}"
                    )
                    progress.update(
                        total_task,
                        description=f"[cyan]总体进度 ({subject_idx + 1}/{total_subjects}) | 平均每Epoch: {avg_epoch_time:.2f}s"
                    )
                    last_update_time = current_time
                
                #### best result
                temp_best_each_subject_pred = []
                if best_accuracy_for_each_subject <= val_acc:
                    best_accuracy_for_each_subject = val_acc
                    temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                    best_each_subject_pred = temp_best_each_subject_pred
                    # Save Weights
                    if (config.train):
                        # torch.save(model.state_dict(), weight_path)
                        best_weights = copy.deepcopy(model.state_dict())
                        

        torch.save(best_weights, weight_path)
        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y.tolist()
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)


if __name__ == '__main__':
    # get_whole_u_v_os()
    # create_norm_u_v_os_train_test()
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train', type=strtobool, default=False)  # Train or use pre-trained weight for prediction
    config = parser.parse_args()
    main(config)
