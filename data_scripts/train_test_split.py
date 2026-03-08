import re
import numpy as np
# Function to separate the contents of a file based on a specific pattern
def separate_file(input_file_path):
    # Regular expression pattern to match lines like "N------>M"
    # where N is any number of digits followed by dashes and then any number
    pattern = re.compile(r'^(\d+)-+>(\d+)$')
    flag=0
    count=0
    # Open the input file for reading
    with open(input_file_path, 'r') as file:
        # Open two files for writing: one for matches, one for others
        #with open('/s/chromatin/c/nobackup/deepplant/Data/Train_non_masked_400_70_10kb.txt', 'w') as matches, open('/s/chromatin/c/nobackup/deepplant/Data/Test_non_masked_400_70_10kb.txt', 'w') as non_matches:
            # Read through each line in the input file
            next(file)
            list1=[]
            temp_list=[]
            for line in file:

                match = pattern.match(line.strip())
                match1=0
                if match:
                    match1=1
                    # Extract the number after the dashes
                    number = int(match.group(2))
                    list1.append(temp_list)
                    temp_list=[]
                    # Check if the number is greater than 3000
                    if number > 3000:
                        flag=1
                    else:
                        flag=0
                if match1==0:
                    #if flag==1:# and count!=1:
                            #matches.write(line)
                            temp_list.append(line)
                    # Write to non_matches if not a match or number <= 3000
                    #else:
                    #    non_matches.write(line)
    return list1
# Example usage
def write_file(list1,name="Train",mask="non_masked",size="10"):
    with open(f'/s/chromatin/c/nobackup/deepplant/Data/{name}_{mask}_400_70_{size}kb.txt', 'w+') as matches:
        for i in range(len(list1)):
            matches.write(list1[i])

def gather_10(temp_list1,train_per,val_per,test_per):
    train_list1=[]
    val_list1=[]
    test_list1=[]
    count1=0
    for k in range(len(temp_list1)):
        if k==1 or count1<train_per :
            #print(f"Inside {len(temp_list1[k])}")
            if train_per<=len(temp_list1[k]):
                train_list1.extend(temp_list1[k][:train_per])
                if count1<=train_per+val_per:
                    val_list1.extend(temp_list1[k][train_per:])
                    count1+=len(temp_list1[k])
            else:
                train_list1.extend(temp_list1[k])
                count1+=len(temp_list1[k])
                        
            
        elif count1<train_per+val_per:
            print(count1,train_per+val_per,len(temp_list1[k]))
            if len(temp_list1[k])>train_per+val_per:
                val_list1.extend(temp_list1[k][:train_per+val_per])
                test_list1.extend(temp_list1[k][train_per+val_per:])
                count1+=len(temp_list1[k])
            else:
                val_list1.extend(temp_list1[k])#[:train_per+val_per])
                
        else:
            test_list1.extend(temp_list1[k])
    return train_list1,val_list1,test_list1
def gather_20(temp_list1,train_per,val_per,test_per):
    train_list1=[]
    val_list1=[]
    test_list1=[]
    count1=0
    temp_list2=[]
    for i in temp_list1:
        for j in i:
    
            temp_list2.append(j)#[i for i in (temp_list1)])
    for k in range(len(temp_list2)):
        if count1<train_per :
            train_list1.append(temp_list2[k])
            count1+=1             
            
        elif count1<train_per+val_per:
            val_list1.append(temp_list2[k])
            count1+=1             
                
        else:
            print(temp_list2[k],len(test_list1),count1,k)
            test_list1.append(temp_list2[k])
            
    return train_list1,val_list1,test_list1
#temp_list1=separate_file('/s/chromatin/b/nobackup/deepplant//Data/arabidopsis_thaliana/rpgc_non_masked_slide200_window_2500_width_200_10kb/labels_non_masked_400_70_10kb.txt')
temp_list2=separate_file('/s/chromatin/b/nobackup/deepplant//Data/arabidopsis_thaliana/rpgc_non_masked_slide200_window_2500_width_200_20kb/labels_non_masked_400_70_20kb.txt')
#temp_list3=separate_file('/s/chromatin/b/nobackup/deepplant//Data/arabidopsis_thaliana/new_rpgc_masked_slide200_window_2500_width_200_10kb/labels_masked_400_70_10kb.txt')
temp_list4=separate_file('/s/chromatin/b/nobackup/deepplant//Data/arabidopsis_thaliana/new_rpgc_masked_slide200_window_2500_width_200_20kb/labels_masked_400_70_20kb.txt')
#len1=np.sum([len(i) for i in temp_list1])
len2=np.sum([len(i) for i in temp_list2])
#len3=np.sum([len(i) for i in temp_list3])
len4=np.sum([len(i) for i in temp_list4])
'''train_per1=int(80*len1/100)
val_per1=int(10*len1/100)
test_per1=len1-(train_per1+val_per1)'''
train_per2=int(80*len2/100)
val_per2=int(10*len2/100)
test_per2=len2-(train_per2+val_per2)
print(len2,train_per2,val_per2,test_per2)
'''train_per3=int(80*len3/100)
val_per3=int(10*len3/100)
test_per3=len3-(train_per3+val_per3)i'''
train_per4=int(80*len4/100)
val_per4=int(10*len4/100)
test_per4=len4-(train_per4+val_per4)
'''count1=0
train_list1,val_list1,test_list1=gather(temp_list1,train_per1,val_per1,test_per1)   
print(f'------>{len(train_list1)},{len(val_list1)},{len(test_list1)}-----"10"    ')
write_file(train_list1,"Train","non_masked","10")     
write_file(val_list1,"Val","non_masked","10")     
write_file(test_list1,"Test","non_masked","10")     

'''



train_list2,val_list2,test_list2=gather_20(temp_list2,train_per2,val_per2,test_per2)        
print(f'------>{len(train_list2)},{len(val_list2)},{len(test_list2)}--------"20"')
print(temp_list2[0],temp_list2[1][:10])
write_file(train_list2,"Train","non_masked","20")     
write_file(val_list2,"Val","non_masked","20")     
write_file(test_list2,"Test","non_masked","20")     

'''



train_list3,val_list3,test_list3=gather(temp_list3,train_per3,val_per3,test_per3)        
print(f'------>{len(train_list3)},{len(val_list3)},{len(test_list3)}--------"10"')
write_file(train_list3,"Train","masked","10")     
write_file(val_list3,"Val","masked","10")     
write_file(test_list3,"Test","masked","10")     


'''

train_list4,val_list4,test_list4=gather_20(temp_list4,train_per4,val_per4,test_per4)        
print(f'------>{len(train_list4)},{len(val_list4)},{len(test_list4)}---------"20"')
write_file(train_list4,"Train","masked","20")     
write_file(val_list4,"Val","masked","20")     
write_file(test_list4,"Test","masked","20")     
