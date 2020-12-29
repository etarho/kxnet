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

# App for interactive camera calibration

# +
import glob

import cv2
import numpy as np
import yaml

# -


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


class Calibrate():
    def __init__(self):
        print("""
    ==============================
      Camera calibration
    ------------------------------
    Grid size
        """)
        self.grid_h = int(input("Grid height: "))
        self.grid_w = int(input("Grid width : "))
        cv2.namedWindow("Capturing", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected grid", cv2.WINDOW_NORMAL)

        self.point3d = np.zeros((self.grid_h * self.grid_w, 3), np.float32)
        self.point3d[:, :2] = np.mgrid[0:self.grid_h, 0:self.grid_w].T.reshape(-1, 2)

        point3d_list = []
        point2d_list = []
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img_list = glob.glob(r'C:\Users\admin\Pictures\Camera Roll\*.jpg')
        assert len(img_list) != 0

        for file in img_list:
            frame = cv2.imread(file)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)
            if ret:
                point3d_list.append(self.point3d)
                point2d_list.append(corners)
                corners_ = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )
                cv2.drawChessboardCorners(frame, (self.grid_h, self.grid_w), corners_, True)
                cv2.imshow("Detected grid", frame)

                print("\rFound (" + str(len(point2d_list)) + " frames)", end="")
            else:
                print("Not found")

            cv2.destroyWindow("Detected grid")
            cv2.namedWindow("Detected grid", cv2.WINDOW_NORMAL)

        print("\n\n-- Calibration --")
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            point3d_list, point2d_list, gray.shape[::-1], None, None
        )
        img_shape = frame.shape
        print(str(len(point2d_list)) + " frames are used.")
        print("Image size :", img_shape)
        print("RMS :", ret)
        print("Intrinsic parameters :")
        print(self.mtx)
        print("Distortion parameters :")
        print(self.dist)

        np.save("camera_config/intrinsic_paramter", self.mtx)
        np.save("camera_config/distortion_paramter", self.dist)

        with open("camera_config/camera_config.yaml", "w") as f:
            yaml.dump(
                {"image_size": {"h": img_shape[0], "w": img_shape[1]}, "rms": ret},
                f,
                default_flow_style=False,
            )

        print("Saved successfully !!\n")

        fs = cv2.FileStorage("camera_config/calibration_result.xml", cv2.FILE_STORAGE_WRITE)
        fs.write("img_shape", img_shape)
        fs.write("rms", ret)
        fs.write("intrinsic", self.mtx)
        fs.write("distortion", self.dist)
        fs.release()
        cv2.destroyAllWindows()

    def test(self):
        print('\nStart testing')
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

        for file in glob.glob("./images/*.jpg"):
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.grid_h, self.grid_w), None)

            if ret:
                corners_ = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.point3d, corners_, self.mtx, self.dist)
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
                img = draw(img, corners_, imgpts)
                cv2.imshow("img", img)
                k = cv2.waitKey(0) & 0xFF
                if k == "s":
                    cv2.imwrite(file[:6] + ".png", img)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrator = Calibrate()
    calibrator.test()

# + active=""
# def calibrate():
#     print(
#         """
#         ==============================
#               Camera calibration
#         ==============================
#
#                 --- Mode ---
#
#         1 : Chessboard pattern
#         2 : Dot grid pattern
#         """
#     )
#
#     while True:
#         s = input("Select pattern to use (1 or 2) :")
#         s.strip()
#         if s[0] == "1":
#             mode = 1
#             print("Chessboard pattern is selected")
#             break
#         elif s[0] == "2":
#             mode = 2
#             print("Dot grid pattern is selected")
#             break
#
#     print("\n-- Grid size --")
#     grid_h = int(input("Grid height: "))
#     grid_w = int(input("Grid width : "))
#
#     print("\n-- Capturing --\nInitializing camera ...")
#     cap = cv2.VideoCapture(0)
#
#     cv2.namedWindow("Capturing", cv2.WINDOW_NORMAL)
#     cv2.namedWindow("Detected grid", cv2.WINDOW_NORMAL)
#
#     print(
#         """
#     -- Commands (enter key while focusing on OpenCV's windows) --
#     <space> : Try to find the grid pattern
#     c       : Cancel previous shot (press if you find detection failure
#     g       : Go to calibration step
#     q       : Quit without calibration
#     """
#     )
#
#     point3d = np.zeros((grid_h * grid_w, 3), np.float32)
#     point3d[:, :2] = np.mgrid[0:grid_h, 0:grid_w].T.reshape(-1, 2)
#
#     point3d_list = []
#     point2d_list = []
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#     cancelable = False
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         else:
#             print("Failed to get frame.")
#             continue
#
#         cv2.imshow("Capturing", gray)
#         key = cv2.waitKey(1)
#         if key == ord("q"):
#             cam.terminate()
#             return
#         elif key == ord("g"):
#             break
#         elif key == ord(" "):
#             if mode == 1:
#                 ret, corners = cv2.findChessboardCorners(
#                     gray, (grid_h, grid_w), None
#                 )
#             elif mode == 2:
#                 ret, corners = cv2.findCirclesGrid(
#                     frame,
#                     (grid_h, grid_w),
#                     None,
#                     cv2.CALIB_CB_CLUSTERING | cv2.CALIB_CB_SYMMETRIC_GRID,
#                 )
#
#             if ret:
#                 point3d_list.append(point3d)
#                 point2d_list.append(corners)
#                 corners_ = cv2.cornerSubPix(
#                     gray, corners, (11, 11), (-1, -1), criteria
#                 )
#                 cv2.drawChessboardCorners(frame, (grid_h, grid_w), corners_, True)
#                 cv2.imshow("Detected grid", frame)
#
#                 cancelable = True
#                 print("\rFound (" + str(len(point2d_list)) + " frames)", end="")
#             else:
#                 print("Not found")
#         elif key == ord("c") and cancelable:
#             point3d_list.pop()
#             point2d_list.pop()
#             cancelable = len(point2d_list) > 0
#             print("Canceled (" + str(len(point2d_list)) + " frames)")
#             cv2.destroyWindow("detected grid")
#             cv2.namedWindow("detected grid", cv2.WINDOW_NORMAL)
#
#     print("\n\n-- Calibration --")
#     ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.calibrateCamera(
#         point3d_list, point2d_list, gray.shape[::-1], None, None
#     )
#     img_shape = frame.shape
#     print(str(len(point2d_list)) + " frames are used.")
#     print("Image size :", img_shape)
#     print("RMS :", ret)
#     print("Intrinsic parameters :")
#     print(cam_mtx)
#     print("Distortion parameters :")
#     print(cam_dist)
#
#     np.save("camera_config/intrinsic_paramter", cam_mtx)
#     np.save("camera_config/distortion_paramter", cam_dist)
#
#     with open("camera_config/camera_config.yaml", "w") as f:
#         yaml.dump(
#             {"image_size": {"h": img_shape[0], "w": img_shape[1]}, "rms": ret},
#             f,
#             default_flow_style=False,
#         )
#
#     print("Saved successfully !!\n")
#
#     fs = cv2.FileStorage("camera_config/calibration_result.xml", cv2.FILE_STORAGE_WRITE)
#     fs.write("img_shape", img_shape)
#     fs.write("rms", ret)
#     fs.write("intrinsic", cam_mtx)
#     fs.write("distortion", cam_dist)
#     fs.release()
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return grid_h, grid_w, cam_mtx, cam_dist
# -


# ##
