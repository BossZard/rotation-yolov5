the rotation detection
# Requirement
```bash
torch==1.6
shapely==1.7.1
opencv==4.2.0.34
```
# inference
you can download the [weights](https://pan.baidu.com/s/1l7AwoT78tQEQ-K_vOJobQQ)(password is 4ud5) for ship detection by my dataset(not DOTA) to test the demo.
```bash
$ python detect.py
```
![image](result.png)
# train
## what format my model need
Not much different from yolo dataset,just add an __angle__ and we define the box attribute w is always __longer__ than h!

So wo define the box label is (cls, c_x, c_y, Longest side，short side, angle)

Attention!we define angle is a classify question,so we define 180 classes for angle.

For Example:
![image](rbox.png)
Range for angle is [-90，90), so wo should __add__ __90__ in angle while make your dataset label and then your label's Range should be [0,179)
![image](label_format.png)
## modify yaml
models/yolov5m.yaml: set nc to your dataset class num;
data/wheat0.yaml:set nc to your dataset class num, and set names to your dataset class name;

```bash
$ python train.py
```
# update
2021.1.4---correct some BUG for training


# details
If you have any question,welcome discuss with me by [This](https://zhuanlan.zhihu.com/p/270388743) or email to prozacliang@qq.com
