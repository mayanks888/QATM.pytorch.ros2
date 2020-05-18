import pandas as pd
import numpy as np
# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
data = pd.read_csv("xshui.csv")
# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
print(data.head())
# mydata = data.groupby('class')
# print(data.groupby('class').count())
# index = mydata.groups['car'].values



# pyindex=1
# data.drop(data.index[pyindex],inplace=True)
# df = pd.DataFrame.drop('1002')
# df = data.drop('car',axis=1,columns='class')
# data.drop('1002',axis=1,columns='obj_id')

df=data
#to remove unwanted id from csv (remove whole row with given column id )
# df=(df[df['obj_id'] != 1002])
#################################3+
xpos_diff=df['x_pos'].diff(periods=-1).abs()
ypos_diff=df['y_pos'].diff(periods=-1).abs()
cool=(df['x_pos'].diff(periods=-1).abs()+df['y_pos'].diff(periods=-1).abs())
total_diff=xpos_diff+ypos_diff
# df=(df[(df['x_pos'].diff(periods=-1).abs()) >=.1])
# df=(df[(df['x_pos'].diff(periods=-1).abs()>=.1) | (df['x_pos'].diff(periods=-1).abs()==0)])
######################
df['gps_diff'] = total_diff
#to remove unwanted id from csv 
df=(df[df['obj_id'] != 1002])

        #################################3
# mydata = data.groupby('img_name')
# grouped = df.groupby('img_name', axis='rows')
grouped = df.groupby('img_name',sort=True)
cool=df.groupby('img_name',sort=True)['obj_id']
# mydata.groups['img_name'].values
mygroup = df.groupby(['img_name', 'obj_id'],sort=True)
# return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
mydata = grouped.groups
for cool in grouped:
    print(cool)
    for val in cool[1]:
        print(val)
        print(val.values)

for da in mydata.keys():
    index = grouped.groups[da].values
    for read_index in index():
        print(index)
        print(da)
        break

################################33
pd1 = pd.DataFrame(df)
pd1.to_csv('output1.csv',index=False)
print(1)
