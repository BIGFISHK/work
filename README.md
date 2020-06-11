## 1+x 中级案例 （已完成）
输出为案例手册。包括手势识别，车牌识别，口罩识别三个zip文件。依次对应深度神经网络，卷积神经网络，迁移学习部分。

## 1+x 中级试题 （已完成）
深度学习试题.xlsx 内为客观题部分。中级操作题.doc 中为操作题部分。

## 1+x 模型部署部分 （部分完成）
完成代码部分，尚未编写手册。项目在**部署**文件夹中。需要配置环境，安装Paddle, opencv, flask。执行项目中app.py文件即可运行，默认部署在本地，需要修改服务器地址修改application.run()中参数。

## 1+x 深度学习教案部分（进行中）
设想是将案例分成三块来写。目前完成了深度学习部分大体框架及部分内容。迁移学习中列了想写的知识点及步骤。

## 试卷分析项目 （部分代码完成）
一开始理解错需求。已完成自动判断试题题型代码部分，是通过BERT模型做了文本的单分类，也完成了本地化部署。项目在**exam**文件夹中。
需要安装Tensorflow-1.13.1, flask
需要从网上下载bert模型以及**chinese_L-12_H-768_A-12**文件夹，两者比较大，没有上传，网上有开源资源。训练文件格式仿照**input**文件夹中
*test*文件。训练样本比较少，主要完成验证，想使用需要重新采集数据训练, 运行*train.py*。
