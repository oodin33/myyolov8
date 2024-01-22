#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     test.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path

from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")


# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection",'Pose', 'Seg', ]

)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        [
            "yolov8n.pt",
            # "yolov8s.pt",
            # "yolov8m.pt",
            # "yolov8l.pt",
            # "yolov8x.pt"
        ]
    )
    model_path = Path(r'weights\detection', str(model_type))

elif task_type == "Cls":
    model_type = st.sidebar.selectbox(
        "Select Model",
        [
            "yolov8n-cls.pt",
            # "yolov8s-cls.pt",
            # "yolov8m-cls.pt",
            # "yolov8l-cls.pt",
            # "yolov8x-cls.pt"
        ]
    )
    model_path = Path(r'weights\cls', str(model_type))
    st.write(model_path)

elif task_type == "Seg":
    model_type = st.sidebar.selectbox(
        "Select Model",
        [
            "yolov8n-seg.pt",
            # "yolov8s-seg.pt",
            # "yolov8m-seg.pt",
            # "yolov8l-seg.pt",
            # "yolov8x-seg.pt"
        ]
    )
    model_path = Path(r'weights\seg', str(model_type))

elif task_type == "Pose":
    model_type = st.sidebar.selectbox(
        "Select Model",
        [
            "yolov8n-pose.pt",
            # "yolov8s-pose.pt",
            # "yolov8m-pose.pt",
            # "yolov8l-pose.pt",
            # "yolov8x-pose.pt"
        ]
    )
    model_path = Path(r'weights\pose', str(model_type))

elif task_type == "Obb":
    model_type = st.sidebar.selectbox(
        "Select Model",
        [
            "yolov8n-obb.pt",
            # "yolov8s-pose.pt",
            # "yolov8m-pose.pt",
            # "yolov8l-pose.pt",
            # "yolov8x-pose.pt"
        ]
    )
    model_path = Path(r'weights\obb', str(model_type))
else:
    st.error(f"Currently only {task_type} function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

# model_path = ""
# if model_type:
#     model_path = Path('weights\detection', str(model_type))
#
# else:
#     st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = YOLO(model_path)

except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Image", "Video", "Webcam"]
)

source_img = None
if source_selectbox == "Image":  # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == "Video":  # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == "Webcam":  # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
