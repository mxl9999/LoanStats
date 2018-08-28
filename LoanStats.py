
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
loandata = pd.DataFrame(pd.read_excel('D:\data analysis\Data capture\loandata.xlsx'))


# In[11]:


loandata


# In[31]:


loandata.duplicated().value_counts()


# In[13]:


loandata.drop_duplicates()


# In[14]:


loandata['loan_amnt'].isnull().value_counts()


# In[15]:


loandata['annual_inc'].isnull().value_counts()


# In[16]:


loandata['loan_amnt'] = loandata['loan_amnt'].fillna(loandata['total_pymnt'] - loandata['total_rec_int']).astype(np.int64)


# In[18]:


loandata['annual_inc'] = loandata['annual_inc'].fillna(loandata['annual_inc'].mean())


# In[20]:


loandata['annual_inc']


# In[21]:


loandata['loan_status'].value_counts()


# In[27]:


loandata['emp_length'].apply(lambda x : x.isalpha())


# In[28]:


loandata['emp_length'].apply(lambda x : x.isalnum())


# In[29]:


loandata['emp_length'].apply(lambda x : x.isdigit())


# In[30]:


loandata.describe().astype(np.int64).T


# In[35]:


import numpy as np
import pandas as pd
loandata = pd.DataFrame(pd.read_excel('D:\data analysis\Data capture\loandata.xlsx'))


# In[36]:


loandata


# In[53]:


loandata.duplicated().value_counts()


# In[54]:


loandata.drop_duplicates()


# In[41]:


loandata.duplicated().value_counts()


# In[55]:


loandata


# In[44]:


loandata.drop_duplicates()


# In[45]:


loandata


# In[46]:


loandata['loan_amnt'].isnull().value_counts()


# In[47]:


loandata['annual_inc'].isnull().value_counts()


# In[58]:


loandata['loan_amnt']=loandata['loan_amnt'].fillna(loandata['total_pymnt']-loandata['total_rec_int'])


# In[59]:


loandata['annual_inc'] = loandata['annual_inc'].fillna(loandata['annual_inc'].mean())


# In[60]:


loandata['loan_status'].value_counts()


# In[65]:


loandata.describe().T


# In[66]:


loandata.replace([3.500000e+04,500.0],loandata['loan_amnt'].mean())


# In[67]:


loandata['loan_amnt'] = loandata['loan_amnt'].astype(np.int64)


# In[69]:


loandata['issue_d'] = pd.to_datetime(loandata['issue_d'])


# In[70]:


loandata.dtypes


# In[71]:


bins = [0,5,10,15,20]
group_names = ['A','B','C','D']
loandata['categories'] = pd.cut(loandata['open_acc'],bins,labels = group_names)


# In[72]:


loandata


# In[73]:


loandata = loandata.set_index('member_id')


# In[74]:


import numpy as np
import pandas as dp
loandata = pd.DataFrame(pd.read_excel('D:\data analysis\Data capture\loandata.xlsx'))


# In[75]:


loandata = loandata.set_index('member_id')


# In[77]:


loandata.head()


# In[78]:


loandata.ix[41000]


# In[79]:


loandata.ix[:,'emp_length']


# In[81]:


loandata.ix[41000,'emp_length']


# In[82]:


loandata.ix[[41000,41001],'loan_amnt']


# In[85]:


loandata.ix[[41000,41001],'loan_amnt'].sum()


# In[86]:


loandata.ix[41000,['loan_amnt','annual_inc']]


# In[87]:


loandata.ix[41000,['loan_data','annual_inc']].sum()


# In[88]:


loandata = loandata.set_index('issue_d')


# In[89]:


loandata


# In[93]:


loandata['2018']


# In[99]:


loandata['2018-03':'2018-05']


# In[101]:


loandata.resample('W',how=sum).head(10)


# In[102]:


loandata.resample('M',how = sum)


# In[103]:


loandata.resample('Q',how = sum)


# In[104]:


loandata.resample('A',how = sum)


# In[105]:


loandata['loan_amnt'].resample('M',how = sum).fillna(0)


# In[106]:


loandata[['loan_amnt','total_rec_int']].resample('M',how = [len,sum])


# In[107]:


loandata['2018-02':'2018-05'].resample('M',how = sum).fillna(0)


# In[108]:


loandata[loandata['loan_amnt'] > 5000].resample('M',how = sum).fillna(0)


# In[114]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
loandata = pd.DataFrame(pd.read_excel('D:\data analysis\Data capture\loandata.xlsx'))


# In[115]:


loandata = loandata.set_index('issue_d')


# In[112]:


loandata


# In[116]:


loan_plot = loandata['loan_amnt'].resample('M').fillna(0)


# In[117]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
loandata = pd.DataFrame(pd.read_excel('D:\data analysis\Data capture\loandata.xlsx'))


# In[119]:


loandata = loandata.set_index('issue_d')


# In[121]:


loandata


# In[122]:


loan_plot = loandata['loan_amnt'].resample('M').fillna(0)


# In[123]:


loan_plot = loandata['loan_amnt'].resample('M').fillna(0)


# In[124]:


loan_grade = loandata.groupby('grade')['loan_amnt'].agg(sum)


# In[125]:


plt.rc('font',family = 'STXihei',size = 15)


# In[126]:


a = np.array([1,2,3,4,5,6])


# In[133]:


plt.bar([1,2,3,4,5,6,7],loan_grade,color='#99CC01',alpha=0.8,align='center',edgecolor='white')
plt.xlabel('用户等级')
plt.ylabel('贷款金额')
plt.title('不同用户等级的贷款金额分布')
plt.legend(['贷款金额'],loc = 'upper right')
plt.grid(color = '#95a5a6',linestyle = '--',linewidth = 1,axis = 'y',alpha=0.4)
plt.xticks(a,('A级','B级','C级','D级','E级','F级','G级'))
plt.show()


# In[136]:


#图表字体为华文细黑，字号为15
plt.rc('font', family='STXihei', size=15)
#创建一个一维数组赋值给a
a=np.array([1,2,3,4,5,6])
#创建条形图，数据源为分等级贷款金额汇总，设置颜色，透明度和图表边框
plt.barh([1,2,3,4,5,6,7],loan_grade,color='#99CC01',alpha=0.8,align='center',edgecolor='white')
plt.xlabel('贷款金额')
plt.ylabel('用户等级')
plt.title('不同用户等级的贷款金额分布')
#添加图例，并设置在图表中的显示位置
plt.legend(['贷款金额'], loc='upper right')
#设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='y',alpha=0.4)
#设置数据分类名称
plt.yticks(a,('A级','B级','C级','D级','E级','F级','G级'))
#显示图表
plt.show()


# In[142]:


#图表字体为华文细黑，字号为15
plt.rc('font', family='STXihei', size=15)
#设置饼图中每个数据分类的颜色
colors = ["#99CC01","#FFFF01","#0000FE","#FE0000","#A6A6A6","#D9E021"]
#设置饼图中每个数据分类的名称
name=['A级', 'B级', 'C级', 'D级', 'E级','F级','G级']
#创建饼图，设置分类标签，颜色和图表起始位置等
plt.pie(loan_grade,labels=name,colors=colors,explode=(0, 0, 0.15, 0, 0, 0,0),startangle=60,autopct='%1.1f%%')
#添加图表标题
plt.title('不同用户等级的贷款金额占比')
#添加图例，并设置显示位置
plt.legend(['A级','B级','C级','D级','E级','F级','G级'], loc='upper left')
#显示图表
plt.show()


# In[143]:


#按月汇总贷款金额，以0填充空值
loan_x=loandata['loan_amnt'].resample('M',how=sum).fillna(0)
#按月汇总利息金额，以0填充空值
loan_y=loandata['total_rec_int'].resample('M',how=sum).fillna(0)
#图表字体为华文细黑，字号为15
plt.rc('font', family='STXihei', size=15)
#创建散点图，贷款金额为x，利息金额为y，设置颜色，标记点样式和透明度等
plt.scatter(loan_x,loan_y,60,color='white',marker='o',edgecolors='#0D8ECF',linewidth=3,alpha=0.8)
#添加x轴标题
plt.xlabel('贷款金额')
#添加y轴标题
plt.ylabel('利息收入')
#添加图表标题
plt.title('贷款金额与利息收入')
#设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
#显示图表
plt.show()


# In[145]:


#按月汇总贷款金额及利息
loan_x=loandata['loan_amnt'].resample('M',how=sum).fillna(0)
loan_y=loandata['total_rec_int'].resample('M',how=sum).fillna(0)
loan_z=loandata['total_rec_int'].resample('M',how=sum).fillna(0)
#图表字体为华文细黑，字号为15
plt.rc('font', family='STXihei', size=15)
#设置气泡图颜色
colors = ["#99CC01","#FFFF01","#0000FE","#FE0000","#A6A6A6","#D9E021",'#FFF16E','#0D8ECF','#FA4D3D','#D2D2D2','#FFDE45','#9b59b6']
#创建气泡图贷款金额为x，利息金额为y，同时设置利息金额为气泡大小，并设置颜色透明度等。
plt.scatter(loan_x,loan_y,s=loan_z,color=colors,alpha=0.6)
#添加x轴标题
plt.xlabel('贷款金额')
#添加y轴标题
plt.ylabel('利息收入')
#添加图表标题
plt.title('贷款金额与利息收入')
#设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
#显示图表
plt.show()


# In[146]:


#图表字体为华文细黑，字号为15
plt.rc('font', family='STXihei', size=15)
#创建箱线图，数据源为贷款来源，设置横向显示
plt.boxplot(loandata['loan_amnt'],1,'rs',vert=False)
#添加x轴标题
plt.xlabel('贷款金额')
#添加图表标题
plt.title('贷款金额分布')
#设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
#显示图表
plt.show()


# In[147]:


#图表字体为华文细黑，字号为15
plt.rc('font', family='STXihei', size=15)
#创建直方图，数据源为贷款金额，将数据分为8等份显示，设置颜色和显示方式，透明度等
plt.hist(loandata['loan_amnt'],8,normed=1, histtype='stepfilled',facecolor='#99CC01', rwidth=0.9,alpha=0.6,edgecolor='white')
#添加x轴标题
plt.xlabel('贷款金额')
#添加y轴标题
plt.ylabel('概率')
#添加图表标题
plt.title('贷款金额概率密度')
#设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='y',alpha=0.4)
#显示图表
plt.show()


# In[148]:


#导入机器学习KNN分析库
from sklearn.neighbors import KNeighborsClassifier
#导入交叉验证库
from sklearn import cross_validation
#导入数值计算库
import numpy as np
#导入科学计算库
import pandas as pd
#导入图表库
import matplotlib.pyplot as plt


# In[149]:


#读取并创建名为knn_data的数据表
knn_data=pd.DataFrame(pd.read_excel('D:\data analysis\Data capture\knn_data.xlsx'))


# In[150]:


#查看数据表前10行
knn_data.head(10)


# In[151]:


#Fully Paid数据集的x1
fully_paid_loan=knn_data.loc[(knn_data["loan_status"] == "Fully Paid"),["loan_amnt"]]
#Fully Paid数据集的y1
fully_paid_annual=knn_data.loc[(knn_data["loan_status"] == "Fully Paid"),["annual_inc"]]
#Charge Off数据集的x2
charged_off_loan=knn_data.loc[(knn_data["loan_status"] == "Charged Off"),["loan_amnt"]]
#Charge Off数据集的y2
charged_off_annual=knn_data.loc[(knn_data["loan_status"] == "Charged Off"),["annual_inc"]]


# In[152]:


#设置图表字体为华文细黑，字号15
plt.rc('font', family='STXihei', size=15)
#绘制散点图，Fully Paid数据集贷款金额x1，用户年收入y1，设置颜色，标记点样式和透明度等参数
plt.scatter(fully_paid_loan,fully_paid_annual,color='#9b59b6',marker='^',s=60)
#绘制散点图，Charge Off数据集贷款金额x2，用户年收入y2，设置颜色，标记点样式和透明度等参数
plt.scatter(charged_off_loan,charged_off_annual,color='#3498db',marker='o',s=60)
#添加图例，显示位置右上角
plt.legend(['Fully Paid', 'Charged Off'], loc='upper right')
#添加x轴标题
plt.xlabel('贷款金额')
#添加y轴标题
plt.ylabel('用户收入')
#添加图表标题
plt.title('贷款金额与用户收入')
#设置背景网格线颜色，样式，尺寸和透明度
plt.grid( linestyle='--', linewidth=0.2)
#显示图表
plt.show()


# In[153]:


#将贷款金额和用户收入设为自变量X
X = np.array(knn_data[['loan_amnt','annual_inc']])
#将贷款状态设为因变量Y
Y = np.array(knn_data['loan_status'])


# In[154]:


#查看自变量和因变量的行数
X.shape,Y.shape


# In[155]:


#将原始数据通过随机方式分割为训练集和测试集，其中测试集占比为40%
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)


# In[156]:


#查看训练集数据的行数
X_train.shape,y_train.shape


# In[157]:


#将训练集代入到KNN模型中
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)


# In[158]:


#使用测试集衡量模型准确度
clf.score(X_test, y_test)


# In[159]:


#设置新数据，贷款金额5000，用户收入40000
new_data = np.array([[5000,40000]])


# In[160]:


#对新数据进行分类预测
clf.predict(new_data)


# In[161]:


#新数据属于每一个分类的概率
clf.classes_,clf.predict_proba(new_data)

