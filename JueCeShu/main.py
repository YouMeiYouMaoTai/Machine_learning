import ID3
import C45
import Date_Create

featLabels1 = []
featLabels2 = []
listWm1, labels1 = Date_Create.createDataSet()   # 构建数据集
listWm2, labels2 = Date_Create.createDataSet()   # 构建数据集
Trees_ID3, featLabels_ID3 = ID3.createTree(listWm1, labels1, featLabels1)   # 生成树
Trees_C45, featLabels_C45 = C45.createTree(listWm2, labels2, featLabels2)   # 生成树
print("\nID3 算法决策树为：", Trees_ID3)
print("C45 算法决策树为：", Trees_C45)
WaterMelon = ['青绿', '硬挺', '沉闷', '清晰', '稍凹', '软粘']
Result_ID3 = ID3.classify(Trees_ID3, featLabels_ID3, WaterMelon)
Result_C45 = C45.classify(Trees_C45, featLabels_C45, WaterMelon)

print("\nID3 算法的预测结果为：")
if Result_ID3 == "是":
    print("好瓜！")
elif Result_ID3 == "否":
    print("歪瓜~")
else:
    print("404---无法预测！")
print("C45 算法的预测结果为：")
if Result_C45 == "是":
    print("好瓜！")
elif Result_C45 == "否":
    print("歪瓜~")
else:
    print("404---无法预测！")

print("SKlearn_ID3的预测结果是：\n歪瓜")
print("SKlearn_C45的预测结果是：\n歪瓜")