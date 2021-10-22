import os
import time
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../", "trainer/"))
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    colorstr,
    increment_path,
    is_ascii,
    non_max_suppression,
    save_one_box,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, colors
from utils.torch_utils import (
    select_device,
    time_sync,
)


def read_imagefile(data) -> Image.Image:
    img_stream = BytesIO(data)
    img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    return img


image_dir = os.path.join(os.path.dirname(__file__), "../../", "images")


@torch.no_grad()
def run_inference(
    pt,
    stride,
    model,
    weights,  # model.pt path(s)
    source,  # file/dir/URL/glob, 0 for webcam
    imgsz=[640],  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="runs",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
):
    imgsz *= 2 if len(imgsz) == 1 else 1
    save_img = not nosave and not source.endswith(".txt")  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    # Run inference
    model(
        torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters()))
    )  # run once
    t0 = time.time()
    for path, img, im0s, _ in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()

        visualize = (
            increment_path(save_dir / Path(path).stem, mkdir=True)
            if visualize
            else False
        )
        pred = model(img, augment=augment, visualize=visualize)[0]
        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        t2 = time_sync()
        res_str = ""
        # Process predictions
        for _, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(
                save_dir / ("".join(p.name.split(".")[:-1]) + ".jpg")
            )  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                res_str += s
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(names[int(cls)]) # cls names
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (names[int(cls)], *xywh, conf)
                            if save_conf
                            else (cls, *xywh)
                        )  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(
                                xyxy,
                                imc,
                                file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                                BGR=True,
                            )

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s)")

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f"Done. ({time.time() - t0:.3f}s)")
    return im0, res_str, save_dir


def predict(pt, stride, model, bytes, iou_thres: float, conf_thres: float, weights):
    img = read_imagefile(bytes.file.read())
    filename = bytes.filename
    filename_path = os.path.join(f"{image_dir}/{filename}")
    cv2.imwrite(filename_path, img)

    _, res_str, save_dir = run_inference(
        pt,
        stride,
        model,
        weights=weights,
        source=filename_path,
        line_thickness=1,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        save_crop=True,
        save_conf=True,
    )
    res_str = res_str[7:].strip()[:-1]
    cropped_img_names = list(Path(save_dir).glob("crops/*/*"))
    return (
        str(res_str),
        (save_dir, ("".join(filename.split(".")[:-1]) + ".jpg")),
        cropped_img_names,
    )
