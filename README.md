# MA-MIDN

Deep learning approaches have demonstrated significant progress in breast cancer histopathological image diagnosis. However, training an interpretative diagnosis model using high-resolution histopathological image is challenging. To alleviate this problem, we propose a novel multi-view attention-guided multiple instance detection network (**MA-MIDN**). We first divide each histopathology image into instances and form a corresponding bag to fully utilize high-resolution information through multiple instance learning (MIL). Then a new multiple-view attention (MVA) algorithm is proposed to learn diverse attention on the instances. A MVA-based MIL pooling strategy is designed for aggregating instance-level features to obtain bag-level features. The proposed MA-MIDN model performs lesion localization and image classification, simultaneously. Particularly, we train the MA-MIDN model under the deep mutual learning (DML) schema. This transfers DML to a weakly supervised learning problem.The proposed method achieves better localization results without compromising classification accuracy.

# MA-MIDN Model
**The MA_MIDN model schematic**
![MA_MIDN.png](https://i.loli.net/2021/04/26/ifBePrOJHXxczpR.png)

# Localization Results
**Here are the localization results generated by the MA-MIDN model on three data sets**
![localization](https://i.loli.net/2021/04/26/QcBrnUohlLdqEvH.png)

# Visualization results generated by Crad-CAM
**The localization results of Crad-CAM on the histopathological image are not very good.**
![grad-cam1](https://i.loli.net/2021/04/26/wIWfYXPeT9yHS7c.png)
![grad-cam2](https://i.loli.net/2021/04/26/5IUwX4uHjY8bR2A.png)


# More Results 
**More  Results**

![](https://i.loli.net/2021/04/29/8GA1jkiQNswCdqx.jpg)  ![](https://i.loli.net/2021/04/29/JEsTbB9lY4RWwDc.jpg)

![](https://i.loli.net/2021/04/29/qdDLY6OVAiek9Qw.jpg)
![](https://i.loli.net/2021/04/29/XeopH36FUu2Yg9a.jpg)
![](https://i.loli.net/2021/04/29/VD3Nqfijbc9gkuY.jpg)
![](https://i.loli.net/2021/04/29/hI5b2ufOHkUma7F.jpg)
![](https://i.loli.net/2021/04/29/KDUYC4oXf16bxF5.jpg)
![](https://i.loli.net/2021/04/29/7nYFTXAKvjrimES.jpg)


