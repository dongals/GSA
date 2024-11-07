# GSA

GaussianAvatar 코드 수정한 것들 여기다 올림. original repo: https://github.com/aipixel/GaussianAvatar?tab=readme-ov-file

## pre-process


## train
    data/${subject}
     ├── images
     ├── masks
     ├── masked_images
     ├── test
     ├── train
     ├── cameras.npz
     └── poses_optimized.npz
.

     data/${subject}/train
     ├── images
     ├── inp_map
     ├── masks
     ├── norm_obj
     ├── cam_parms.npz
     ├── cano_smpl.obj
     ├── query_posemap_256_cano_smpl.npz
     ├── smpl_cano_joint_mat.pth
     ├── smpl_parms.pth
     └── smpl_parms_pred.pth

data-preprocessing 전에는 train 폴더에서 inp_map, norm_obj, smpl_parms_pred.pth가 없음.
 
train stage1을 돌리고 export_stage1_smpl.py를 실행하면 stage1의 결과인 smpl_parms_pred.pth가 나옴. (pose와 transl이 optimize한 것)

 
gen_pose_map_our_smpl.py를 돌리면 norm_obj, inp_map 폴더가 나옴.

부연설명: 수정된 포즈(=smpl_parms_pred.pth)를 smpl 함수에 대입해서 나온 게 norm_obj이고, norm_obj를 pos_render라는 함수를 사용하여 uv_pos, uv_face, face_id parameter를 얻고 inp_posemap_128_*.npz로 저장함.

 
train stage2를 돌릴 때의 차이점은 smpl_parms_pth 대신 smpl_parms_pred.pth를 사용하고, inp_map을 사용하는 것이다.
     
부연설명: inp_map은 pose_encoder를 통과시켜서 gaussian parameter인 point,color,scale을 얻는데 사용한다.

 
이렇게 해서 얻은 train_stage2의 prediction 결과를 렌더링한 것이 가장 정확하고, novel pose를 적용해서 렌더링하면 정확도가 떨어진다.
 
     python train.py -s $path_to_data/$subject -m output/{$subject}_stage1 --train_stage 1 --pose_op_start_iter 10
.

     cd scripts & python export_stage_1_smpl.py

export_stage_1_smpl.py 돌릴 때 path 수정: 

    net_save_path = '/intern1/mmai08/GaussianAvatar/output/dongals/train_stage1/net/iteration_180'
    smpl_parms_path = '/intern1/mmai08/GaussianAvatar/data/user/dongals/train'

     python gen_pose_map_our_smpl.py

gen_pose_map_our_smpl.py 돌릴 때 path 수정: 

    smpl_parm_path = '../data/user/dongals/train'
    parms_name = 'smpl_parms_pred.pth
.

     cd .. &  python train.py -s $path_to_data/$subject -m output/{$subject}_stage2 --train_stage 2 --stage1_out_path $path_to_stage1_net_save_path


## render_novel_pose.py

gen_pose_map_our_smpl.py를 돌려서 novel_pose의 inp_map을 얻어야 함.
path 수정할 것: 

    smpl_parm_path = '../assets/test_pose'
    parms_name = 'smpl_parms.pth'
터미널 명령어:

     python gen_pose_map_our_smpl.py

stage1에서 렌더링하는 경우:

render_novel_pose.py의 line 16:

    avatarmodel.load(epoch)

혹은 avatarmodel.stage_load(epoch)으로 해도 될 것 같음.

설명: assets/test_pose 폴더의 pose정보와 transl 정보를 가져와서 self.net을 통과하여 Gaussian parameter(points, colors, scales)를 얻음.


stage2에서 렌더링하는 경우:

render_novel_pose.py의 line 16:

    avatarmodel.stage2_load(epoch)

avatar_model.py의 render_free_stage2함수를 수정: (아래 코드 부분만 복붙하면 될 것 같습니다.)

    def render_free_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map'] 
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)            
        transl_batch = self.transl(idx)
        pose_data = batch_data['pose_data']     # 여기 추가
        transl_data = batch_data['transl_data'] # 여기 추가

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            #live_smpl = self.smpl_model.forward(betas=self.betas,
            #                    global_orient=pose_batch[:, :3],
            #                    transl = transl_batch,
            #                    body_pose=pose_batch[:, 3:])
            live_smpl = self.smpl_model.forward(betas=self.betas,  # 여기 추가
                                global_orient=pose_data[:, :3],    # 여기 추가
                                transl = transl_data,              # 여기 추가
                                body_pose=pose_data[:, 3:])        # 여기 추가

  아래는 똑같음.
  
  render_free_stage1과의 차이점은 stage2_load함수를 사용하여 self.net, self.geo_geature가 다르고, self.pose_encoder를 사용한다는 점입니다. 
  beta는 그대로인 것 같습니다. 개인적으로 pose_encoder가 큰 역할을 하는지는 모르겠습니다. self.net이 바뀐 게 가장 큰 것 같습니다.
