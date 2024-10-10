"""Image demo script."""
import argparse
import os
import pickle as pk
import time

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

# 입력 비디오 파일의 정보 가져오는 함수 -> 웹캠 대체
#def get_video_info(in_file):
#    stream = cv2.VideoCapture(in_file)
#    assert stream.isOpened(), 'Cannot capture source'
#    # self.path = input_source
#    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
#    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
#    fps = stream.get(cv2.CAP_PROP_FPS)
#    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
#    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
#    stream.release()
#
#    return stream, videoinfo, datalen


#def recognize_video_ext(ext=''):
#    if ext == 'mp4':
#        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
#    elif ext == 'avi':
#        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
#    elif ext == 'mov':
#        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
#    else:
#        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
#        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
# parser.add_argument('--img-path',
#                     help='image name',
#                     default='',
#                     type=str)
#parser.add_argument('--video-name',
#                    help='video name',
#                    default='',
#                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
parser.add_argument('--save-pk', default=False, dest='save_pk',
                    help='save prediction', action='store_true')
parser.add_argument('--save-img', default=False, dest='save_img',
                    help='save prediction', action='store_true')


opt = parser.parse_args()

# 설정 파일 및 모델 초기화
cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet.pth'
cfg = update_config(cfg_file)

# 모델 설정파일 로드&업데이트 -> 설정 객체 생성
bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_uvd',
    'pred_xyz_17',
    'pred_xyz_29',
    'pred_xyz_24_struct',
    'pred_scores',
    'pred_camera',
    # 'f',
    'pred_betas',
    'pred_thetas',
    'pred_phi',
    'pred_cam_root',
    # 'features',
    'transl',
    'transl_camsys',
    'bbox',
    'height',
    'width',
    'img_path'
]
res_db = {k: [] for k in res_keys}

# 입력이미지 변환을 정의하는 객체. 모델이 필요로 하는 입력 형식에 맞게 이미지를 변환
transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

# pretrained Faster R-CNN 모델과 HybrIK 모델을 생성
det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)

# 모델 가중치 로드
print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

# 모델을 GPU에 로드하고 평가모드 설정.
det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

# 출력 디렉토리 설정
print('### Extract Image...')
#video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

# .mp4파일을 입력으로 받아와서 정보를 가져오는 부분. & basename 추출하는 부분.
#_, info, _ = get_video_info(opt.video_name)
#video_basename = os.path.basename(opt.video_name).split('.')[0]
#savepath = f'./{opt.out_dir}/res_{video_basename}.mp4'
#savepath2d = f'./{opt.out_dir}/res_2d_{video_basename}.mp4'
#info['savepath'] = savepath
#info['savepath2d'] = savepath2d

# -> 웹캠 입력을 받기 위해 cv2.VideoCapture(0)으로 실시간 영상 스트림을 열고
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

savepath = f'./{opt.out_dir}/res_webcam.mp4'
savepath2d = f'./{opt.out_dir}/res_2d_webcam.mp4'

fps = cap.get(cv2.CAP_PROP_FPS)
frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# cv2.VideoWriter 객체를 생성하여 프레임을 비디오 파일로 작성할 수 있도록 설정
#write_stream = cv2.VideoWriter(
#    *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
#write2d_stream = cv2.VideoWriter(
#    *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])
#if not write_stream.isOpened():
#    print("Try to use other video encoders...")
#    ext = info['savepath'].split('.')[-1]
#    fourcc, _ext = recognize_video_ext(ext)
#    info['fourcc'] = fourcc
#    info['savepath'] = info['savepath'][:-4] + _ext
#    info['savepath2d'] = info['savepath2d'][:-4] + _ext
#    write_stream = cv2.VideoWriter(
#        *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
#    write2d_stream = cv2.VideoWriter(
#        *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])

# 위 내용 수정 ->
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
write_stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
write2d_stream = cv2.VideoWriter(savepath2d, fourcc, fps, frameSize)


# VideoWriter 정상적으로 열렸는지 확인
assert write_stream.isOpened(), 'Cannot open video for writing'
assert write2d_stream.isOpened(), 'Cannot open video for writing'

# ffmpeg를 사용해 입력 비디오를 각 프레임별 이미지파일로 추출
#os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')

# 프레임별 이미지파일 저장
#files = os.listdir(f'{opt.out_dir}/raw_images')
#files.sort()

#img_path_list = []

#for file in tqdm(files):
#    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

#        img_path = os.path.join(opt.out_dir, 'raw_images', file)
#        img_path_list.append(img_path)

prev_box = None
renderer = None
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

# 모델 추정
print('### Run Model...')
idx = 0
elapsed_times = []
#for img_path in tqdm(img_path_list):
while True:
    # 매 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        print("Webcam이 정상적으로 실행되지 않았습니다.")
        break

    start_time = time.time()
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV에서 사용하는 BGR 형식을 RGB 형식으로 변환

    #dirname = os.path.dirname(img_path)
    #basename = os.path.basename(img_path)

    with torch.no_grad():
        # Run Detection
        #input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu) # 이미지를 텐서로 변환, GPU로 전송
        det_output = det_model([det_input])[0] # 사람탐지모델로부터 탐지된 객체 output

        # 첫 프레임이라면 -> 사람 탐지 & 바운딩박스 얻기
        if prev_box is None:
            tight_bbox = get_one_box(det_output)  # xyxy
            if tight_bbox is None:
                continue
        else: # 이후부턴 이전프레임의 box와의 최대 IoU(최대한 일치하도록)를 갖는 box
            tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy록

        prev_box = tight_bbox

        # Run HybrIK <- 포즈추정모델에 입력
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
        )

        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        print(f"Computation time per frame : {elapsed_time:.4f} seconds.")
        
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()

        # Visualization
        image = input_image.copy()
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        transl_camsys = transl.clone()
        transl_camsys = transl_camsys * 256 / bbox_xywh[2]

        focal = focal / 256 * bbox_xywh[2]

        vertices = pose_output.pred_vertices.detach()

        verts_batch = vertices
        transl_batch = transl

        color_batch = render_mesh(
            vertices=verts_batch, faces=smpl_faces,
            translation=transl_batch,
            focal_length=focal, height=image.shape[0], width=image.shape[1])

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        input_img = image
        alpha = 0.9
        image_vis = alpha * color[:, :, :3] * valid_mask + (
            1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        #각 프레임은 이미지파일로 저장, write_stream을 통해 비디오 파일로 저장
        if opt.save_img:
            idx += 1
            res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, image_vis)
            #cv2.imshow()<- 이렇게 하면 화면에 실시간 결과 출력
        write_stream.write(image_vis)

        # vis 2d <- 관절 point 시각화, 비디오 작성
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        image = input_image.copy()
        bbox_img = vis_2d(image, tight_bbox, pts)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        write2d_stream.write(bbox_img)

        if opt.save_img:
            res_path = os.path.join(
                opt.out_dir, 'res_2d_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, bbox_img)

        # 결과 파일 저장
        if opt.save_pk:
            assert pose_input.shape[0] == 1, 'Only support single batch in력ference for now'

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu(
            ).data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(
                dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(
                dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(
                dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            res_db['pred_uvd'].append(pred_uvd_jts)
            res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            res_db['pred_scores'].append(pred_scores)
            res_db['pred_camera'].append(pred_camera)
            # res_db['f'].append(1000.0)
            res_db['pred_betas'].append(pred_betas)
            res_db['pred_thetas'].append(pred_theta)
            res_db['pred_phi'].append(pred_phi)
            res_db['pred_cam_root'].append(pred_cam_root)
            # res_db['features'].append(img_feat)
            res_db['transl'].append(transl[0].cpu().data.numpy())
            res_db['transl_camsys'].append(transl_camsys[0].cpu().data.numpy())
            res_db['bbox'].append(np.array(bbox))
            res_db['height'].append(img_size[0])
            res_db['width'].append(img_size[1])
            #res_db['img_path'].append(img_path)
            res_db['img_path'].append('webcam_frame')

    # ESC 키 입력 시 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

average_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
print(f"Average Time: {average_time:.4f} seconds.")


if opt.save_pk:
    n_frames = len(res_db['img_path'])
    for k in res_db.keys():
        print(k)
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, 'res.pk'), 'wb') as fid:
        pk.dump(res_db, fid)

cap.release()
write_stream.release()
write2d_stream.release()
cv2.destroyAllWindows()
