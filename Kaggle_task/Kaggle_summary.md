# **Kaggle_Titanic任务综述**

## 数据探索与预处理的步骤方法

1. 数据探索阶段先根据源数据文件找出缺失值特征，发现cabin特征缺失值太多，我就把cabin特征移除了。
2. 再运用pandas的value_counts功能对加载数据进行分析，分析每个特征对幸存者的影响，为后续特征工程的转换做准备,发现性别和年龄和舱数三个特征最好
3. 年龄缺失值用平均值填充,登船站用众数填充
4. 预处理阶段先把性别和登船口用数字代替,parch和sibsp合并成家庭规模,感觉可以直接用二分类特征,就用新特征，是否独行来做特征,年龄和船费数据分布都太杂乱,用cut方法改用年龄段和船费分段做特征

## 训练阶段及性能比较

1. 数据集用2—8划分,通过准确率和召回率分析每个模型的缺陷基本都在召回率,其中随机森林的无论是召回率还是准确率基本都比其他模型表现好
2. !['KNN'](D:/Backup_Folder/KNN_Report.png)
3. !['Logistic'](D:/Backup_Folder/Logistic_Report.png)
4. !['SVM'](D:/Backup_Folder/SVM_Report.png)
5. !['RandomForest'](D:/Backup_Folder/RandomForest_Report.png)

## 超参数调优过程及最佳模型选择

1. 由上面四图可得随机森林表现最好,再运用 Grid Search 方法对不同参数组合求索,最终得出最佳参数组是

   **{ 'criterion': 'entropy'**

   **'max_depth': 15,**

   **'max_features': 'sqrt'**

   **'min_samples_split': 10**

   **'n_estimators': 50 }**

2. !['BestRandomForest'](D:/Backup_Folder/Best_RandomForest_Report.png)

   最终优化完提升在 **3%** 左右
