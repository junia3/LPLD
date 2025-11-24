# LPLD (Low-confidence Pseudo Label Distillation) (ECCV 2024)

<a href="https://arxiv.org/abs/2407.13524">
  <img src="https://img.shields.io/badge/arXiv-2407.13524-0096c7.svg?style=plastic" alt="arXiv">
</a>

This is an official code implementation repository for ```Enhancing Source-Free Domain Adaptive Object Detection with Low-confidence Pseudo Label Distillation```, accepted to ```ECCV 2024```.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0b18f570-7aed-4a05-a205-722fb052b80a">
</p>

---

## Installation and Environmental settings (Instructions)

- We use Python 3.6 and Pytorch 1.9.0
- The codebase from [Detectron2](https://github.com/facebookresearch/detectron2).

```bash
git clone https://github.com/junia3/LPLD.git

conda create -n LPLD python=3.6
conda activate LPLD
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

cd LPLD
pip install -r requirements.txt

## Make sure you have GCC and G++ version <=8.0
cd ..
python -m pip install -e LPLD

```

---

## Dataset preparation
- Cityscapes, FoggyCityscapes / [Download Webpage](https://www.cityscapes-dataset.com/)
- PASCAL_VOC / [Download Webpage](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
- Clipart / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)
- Watercolor / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)
- Sim10k / [Download Webpage](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)

Make sure that all downloaded datasets are located in the ```./dataset``` folder. After preparing the datasets, you will have the following file structure:

```bash
LPLD
...
├── dataset
│   └── foggy
│   └── cityscape
│   └── clipart
│   └── watercolor
...
```
Make sure that all dataset fit the format of PASCAL_VOC. For example, the dataset foggy is stored as follows:

```bash
$ cd ./dataset/foggy/VOC2007/
$ ls
Annotations  ImageSets  JPEGImages
$ cat ImageSets/Main/test_t.txt
target_munster_000157_000019_leftImg8bit_foggy_beta_0.02
target_munster_000124_000019_leftImg8bit_foggy_beta_0.02
target_munster_000110_000019_leftImg8bit_foggy_beta_0.02
.
.
```

---

## Visualize

<p align="center">
  <img src="https://github.com/user-attachments/assets/70769452-c522-4551-b46a-a0fec3d2aaf0", width="400">
  <img src="https://github.com/user-attachments/assets/5b0803c4-3169-4812-b319-169a561b3807", width="400">
</p>

---

## Citation
  
```bibtex
@inproceedings{yoon2024enhancing,
  title={Enhancing Source-Free Domain Adaptive Object Detection with Low-Confidence Pseudo Label Distillation},
  author={Yoon, Ilhoon and Kwon, Hyeongjun and Kim, Jin and Park, Junyoung and Jang, Hyunsung and Sohn, Kwanghoon},
  booktitle={European Conference on Computer Vision},
  pages={337--353},
  year={2024},
  organization={Springer}
}
```
