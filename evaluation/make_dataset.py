# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Generating evaluation dataset
# -

import cv2
import numpy as np
import pandas as pd
from IPython.display import clear_output
from scipy.spatial.transform import Rotation
from sksurgerynditracker.nditracker import NDITracker

ROMFILE_DIR = "./tracking/Slicer/rom_config/"

rom_list = ["custom_001", "chessboard"]

TRACKER_SETTINGS = {
    "tracker type": "polaris",
    "romfiles": list(map(lambda x: ROMFILE_DIR + x + ".rom", rom_list)),
}


class SyncCamTrak:
    def __init__(self):
        print("\nInitializing tracker...")
        self.tracker = NDITracker(TRACKER_SETTINGS)
        self.tracker.start_tracking()

        print("Initializing camera...\n")
        self.cap = cv2.VideoCapture(0)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video = cv2.VideoWriter("eval.mp4", fourcc, fps, (w, h))

    def get_transform(self, tool, frame_num, mat):
        assert mat.shape == (4, 4)
        x = mat[0, 3]
        y = mat[1, 3]
        z = mat[2, 3]
        rot_mat = mat[:3, :3]
        rot = Rotation.from_matrix(rot_mat).as_euler("ZXY")
        alpha, beta, gamma = np.rad2deg(rot)
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = rot_mat.flatten()
        print(
            """
        =================
        ID: {}
        Frame: #{}
        -----------------
        Tx: {:05.2f} mm
        Ty: {:05.2f} mm
        Tz: {:05.2f} mm
        Rx: {:05.2f} deg
        Ry: {:05.2f} deg
        Rz: {:05.2f} deg
        =================
        """.format(
                tool,
                frame_num,
                x,
                y,
                z,
                alpha,
                beta,
                gamma,
            )
        )
        return frame_num, x, y, z, r1, r2, r3, r4, r5, r6, r7, r8, r9

    def on(self):
        j = 0
        tool_dict = {"1": "forceps", "2": "endoscope"}
        transform_dict = {"forceps": [], "endoscope": []}
        while True:
            clear_output(wait=True)
            id_, _, _, mat, _ = self.tracker.get_frame()
            for i in id_:
                tool = tool_dict[str(i)]
                transform = self.get_transform(tool, j, mat[i - 1])
                transform_dict[tool].append(transform)

            ret, frame = self.cap.read()
            cv2.imshow("camera", frame)
            cv2.imwrite("images/img_{:05d}.jpg".format(j+1), frame)
            self.video.write(frame)
            j += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for tool in ["forceps", "endoscope"]:
            df = pd.DataFrame(
                transform_dict[tool],
                columns=[
                    "frame",
                    "x",
                    "y",
                    "z",
                    "r1",
                    "r2",
                    "r3",
                    "r4",
                    "r5",
                    "r6",
                    "r7",
                    "r8",
                    "r9",
                ],
            )
            df.to_csv("{}.csv".format(tool), index=False)
        print("\nFile save successed!!")
        self.cap.release()
        cv2.destroyAllWindows()


main = SyncCamTrak()

input("Press ENTER to start recording\n")

print('Press q to stop recording')

if __name__ == "__main__":
    main.on()


