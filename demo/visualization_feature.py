#coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch

from mmdet.apis import inference_detector, init_detector

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def draw_feature_map(model, img_path, save_dir):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = mmcv.imread(img_path)
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    featuremaps = inference_detector(model, img) #1.这里需要改model，让其在forward的最后return特征图。我这里return的是一个Tensor的tuple，每个Tensor对应一个level上输出的特征图。
    i=0
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 1 + img *0  # 这里的0.4是热力图强度因子
        cv2.imwrite(os.path.join(save_dir,'featuremap_'+str(i)+'.png'), superimposed_img)  # 将图像保存到硬盘
        i=i+1


from argparse import ArgumentParser

# def main():
#     parser = ArgumentParser()
#     parser.add_argument('img', help='Image file')
#     parser.add_argument('save_dir', help='Dir to save heatmap image')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#     parser.add_argument('--device', default='cuda:0', help='Device used for inference')
#     args = parser.parse_args()

#     # build the model from a config file and a checkpoint file
#     model = init_detector(args.config, args.checkpoint, device=args.device)
#     draw_feature_map(model,args.img,args.save_dir)

if __name__ == '__main__':
    # main()

    imgPath = '/home/ubuntu/mmdetection3/demo/test_video/'
    save_dir = '/home/ubuntu/mmdetection3/demo/visfeature_video2/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    config_file = '/home/ubuntu/mmdetection3/work_dirs/cascade_rcnn_r50_fpn_1x_coco1/cascade_rcnn_r50_fpn_1x_coco.py'
    checkpoints = '/home/ubuntu/mmdetection3/work_dirs/cascade_rcnn_r50_fpn_1x_coco1/epoch_2.pth'

    model = init_detector(config_file, checkpoints, device='cuda:0')

    img_list = os.listdir(imgPath)
    
    for i in range(len(img_list)):

        imgname = imgPath + img_list[i]
        print(imgname)

        outname = save_dir + img_list[i][:-4]+'/'
        # print(outname)

        if not os.path.isdir(outname):
            os.makedirs(outname)

        print(outname)

        draw_feature_map(model,imgname,outname)