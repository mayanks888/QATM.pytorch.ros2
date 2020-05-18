

bbox_info=[1,2,3,4]
temp_name=['a','b','c','d']
temp_info={}
temp_info2={}
for i in range(4):

    dict={temp_name[i]:bbox_info[i]}
    temp_info.update(dict)
    temp_info2.update(i,bbox_info[i])

1