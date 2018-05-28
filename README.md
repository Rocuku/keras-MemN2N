# keras-MemN2N

**过程详见blog：**使用 keras 复现 MemN2N[](https://rocuku.github.io/keras-memn2n/)

**参考论文：**[《End-To-End Memory Networks》](https://arxiv.org/abs/1503.08895)

**数据集：**[bAbI-tasks](https://research.fb.com/downloads/babi/)

**最终源码：**[keras-MemN2N（GitHub）](https://github.com/Rocuku/keras-MemN2N)

**前期工作：**

- [《End-To-End Memory Networks》论文阅读笔记](https://rocuku.github.io/End-To-End-Memory-Networks/)
- [Memory Networks 相关论文整理总结](https://rocuku.github.io/memory-networks-summary/)
 
- - -

BoW + Adjacent 效果很差，task-1 只能到 0.66+，但是 task-20 能到 1（数据集 1k 还是 10k 都差不多，hop 取 3 和取 1 也差不多）

