from argparse import ArgumentParser

from pathlib import Path
import numpy as np
import torch

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from util import joints_num, process_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--smpl_params_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="motion_vecs.npy",
    )

    args = parser.parse_args()
    smpl_params_path = Path(args.smpl_params_path)
    output_dir = Path(args.output_dir)
    output_file_name = args.output_file_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    TRANS_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    male_bm_path = "./body_models/smplh/male/model.npz"

    num_betas = 10

    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas).to(device)
    faces = c2c(male_bm.f)

    smpl_params_raw = np.load(smpl_params_path, allow_pickle=True)
    body_parms = {
        "root_orient": torch.Tensor(smpl_params_raw["global_orient"]).to(device),
        "pose_body": torch.Tensor(smpl_params_raw["body_pose"][:, :63]).to(device),
        "pose_hand": torch.zeros(len(smpl_params_raw["transl"]), 90).to(device),
        "trans": torch.Tensor(smpl_params_raw["transl"]).to(device),
        "betas": torch.Tensor(smpl_params_raw["betas"]).to(device),
    }
    with torch.no_grad():
        body = male_bm(**body_parms)
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, TRANS_MATRIX)
    pose_seq_np_n[..., 0] *= -1
    source_data = pose_seq_np_n[:, :joints_num]
    data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
    np.save(output_dir / output_file_name, data)
