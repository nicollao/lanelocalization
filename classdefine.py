import myconfig
import operator

to_classnum = {
'0_00_01_01_00':(0, '1', '1' , 0,  10000),
'0_00_01_02_00':(1, '1', '2' , 1,  10000),
'0_00_01_x3_00':(2, '1', '3x', 2,  10000),

'0_00_02_03_00':(3, '2', '3' , 3,  10000),
'0_00_02_x4_00':(4, '2', '4x', 4,  10000),

'0_00_03_x5_00':(5, '3', '5x', 5,  10000),

'1_00_01_02_00':(6, 'r1', '2', 6,  10000),
'1_00_01_x3_00':(7, 'r1', '3x', 7, 10000),

'1_00_02_x5_00':(8, 'r2', '4x', 8, 10000),
'0_00_03_04_00':(9,  '3', '4',  11,10000),
'1_00_03_x5_00':(10, 'r3','5x', 12,  9000),

'0_00_01_01_01':(11, '1', '1M',  1,    0),
'0_00_01_01_x2':(12, '1', '1M',  2,    0),
'0_00_01_02_x1':(13, '1', '2M',  2, 550),
'0_00_02_03_x1':(14, '2', '3M',  4, 500),

'1_00_01_02_01':(15, 'r1', '1M', 9, 1500),
# '1_00_01_02_x2':(16, 'r2', '2M', 13,  180),
'1_00_01_02_x2':(16, 'r1', '2M', 9,  180),
'1_00_01_x3_01':(17, 'r1', '3xM',9, 900),
# '1_00_01_x3_x2':(18, 'r2', '3xM',13, 990),
'1_00_01_x3_x2':(18, 'r1', '3xM',9, 990),
'1_00_02_x4_x1':(19, 'r2', '4xM',10,2700),

'1_01_01_01_00':(20, 'r1', '1B', 6,    0),
'1_x2_01_01_00':(21, 'r1', '1B', 7, 1100),
'1_x1_01_02_00':(22, 'r1', '2B', 7,  90),
'1_x2_01_02_00':(23, 'r1', '2B', 7,   0),
'1_x2_02_02_00':(24, 'r2', '2B', 8, 450),
'1_x1_01_x3_00':(25, 'r1', '3xB', 7,  0),
'1_x1_02_03_00':(26, 'r2', '3B', 8,    0),
'1_x2_03_03_00':(27, 'r3', '3B', 12,    0),
'1_00_02_04_00':(28, '2', '4',   4,    0),
}

to_simpleclass ={
  '1' : 0,
  '2' : 1 ,
  '3' : 2 ,
  'r1': 3 ,
  'r2': 4 ,
  'r3': 3
}

def classnum_to_type(class_num) :

    for key, value in to_classnum.items() :
        if value[myconfig.class_type] == class_num :
            return  key, value[1], value[2], value[3], value[4]
    return "", "", "", ""

def type_to_classnum(class_num) :

    for key, value in  to_classnum.items() :
        if key == class_num :
            return  value[0], value[1], value[2], value[3], value[4]
    return "", "", "", ""

def midclassnum_to_type(class_num) :

    sorted_x = sorted(to_classnum.items(), key=operator.itemgetter(1))

    for key, value in sorted_x:
        if value[myconfig.class_type] == class_num :
            return  key
    return ""

def simpleclass_to_type(class_num) :

    for key, value in  to_simpleclass.items() :
        if value == class_num :
            return key
    return ''

def type_to_simpleclass(class_num) :

    for key, value in  to_simpleclass.items() :
        if key == class_num :
            return value
        # if class_num == 11 and
    return ''

def type_to_simpleclass1(label1, label2) :

    for key, value in  to_simpleclass.items() :
        if key == label[1]:
            return value
        if key == label[5]:
            return value
    return ''
