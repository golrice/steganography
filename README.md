# steganography

## 环境配置

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## 新增内容

2025.5.31  1:04  xlh

embeding_test文件夹中可以运行embeding.py来对单张图片进行误码率分析，也可以运行dct_test.py来对实现dct变换的函数进行debug

## to_do_1

2025.5.31  1:04  xlh

找到误码产生的原因，尝试优化，image_to_dct，dct_to_image，compute_channel_distortion这三个函数，从而降低误码率

compute_channel_distortion函数所使用的代价函数算法是课程项目任务文档中老师描述的信道失真代价，如实现有误请指出并更改

倘若有更优良的代价函数也可以将该函数整个替换。

embeding.py中没有进行从71到96这二十多次的jpeg压缩，只进行了一次的量化，但是误码率已经达到了50%。我们的目标是把误码率尽量降低到0。

可以先尝试在embeding.py中降低误码率，再将相应的有贡献的函数更改到src里，从而观察大批量数据的误码率是否有所下降。

如有进度更新，请在readme文件中说明

## to_to_2

2025.6.1 golrice

fix 了一些可能的错误(但误码率没有发生显著变化，还在查找问题)

## to_do_3

2025.6.3 xlh

新增了j-uniward失真计算函数和固定失真计算函数，效果没有明显的增益，在jpeg压缩或者带量化的dct变换后BER仍然为50%上下

增加了三种额外的单张图片测试误码率分析代码
