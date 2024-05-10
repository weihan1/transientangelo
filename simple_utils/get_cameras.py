from scipy.spatial.transform import Rotation as R
import numpy as np 
import json 
import os


def txt2json(fname):

    with open(fname) as file:
        lines = [line.rstrip() for line in file]

    def qvec2rotmat(qvec):
        return np.array([
            [
                1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
            ], [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
            ], [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
            ]
        ])

    outpath = os.path.splitext(fname)[0] + '.json'
    out = {"camera_angle_x": 0.690976, "frames": []}
    i = 0 
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    mean_tvec = np.zeros((3))
    for line in lines:
        if line[-3:] == "png":
            line_list = line.split(" ")
            line_list = [float(x) for x in line_list[1:-2]]
            qvec = np.array(line_list[:4])
            tvec = np.array(line_list[-3:])
            # r = R.from_quat(line_list[:4])
            # rot_mat = r.as_matrix()
            # # rot_mat = qvec2rotmat(np.array(line_list[:4]))
            # transvec = np.array(line_list[-3:])
            # c2w[:3, :3] = rot_mat
            # c2w[:3, 3] = transvec

            R = qvec2rotmat(-qvec)
            t = tvec.reshape([3, 1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(m)

            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:]
            c2w[2,:] *= -1 # flip whole world upside down

            frame = {"file_path":f"./train/r_{i}", "rotation": 0.012566370614359171, "transform_matrix": c2w.tolist()}
            out["frames"].append(frame)
            i+= 1
            

    with open(outpath, "w") as outfile:
        json.dump(out, outfile, indent=2)