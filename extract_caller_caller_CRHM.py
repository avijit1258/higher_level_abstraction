import pandas as pd
import numpy as np

fileName = 'CRHMmain.cpp'
infname = 'C:/Users/Saikat Mondal/PycharmProjects/HigerLevelAbstraction/instrumenting source code/original_sources/'+fileName
classNameForDef = 'TMain'

file = []
className = []
functionName = []
numberOfParameters = []



fin = open(infname)

cpp_file_list = ['About.cpp','AKAform.cpp','Analy.cpp','Bld.cpp','ClassCRHM.cpp','ClassModule.cpp','Common.cpp','CRHM_parse.cpp','CRHMmain.cpp' \
                'EntryForm.cpp','Examples.cpp','Export.cpp','Export.cpp','GlobalCommon.cpp','GlobalDll.cpp','Hype_CRHM.cpp','Hype_lake.cpp', \
                'Hype_river.cpp','Hype_routines.cpp','Log.cpp','MacroUnit.cpp','NewModules.cpp','Numerical.cpp','Para.cpp','report.cpp','UpdateForm.cpp']

mappingFileClass = {}

mappingFileClass[cpp_file_list[0]] = ['TAboutBox']
mappingFileClass[cpp_file_list[1]] = ['TFormAKA']
mappingFileClass[cpp_file_list[2]] = ['Plot', 'TAnalysis']
mappingFileClass[cpp_file_list[3]] = ['TBldForm']
#3975 number line of classCRHM.cpp is considered
mappingFileClass[cpp_file_list[4]] = ['ClassVar', 'ClassPar','ClassData','Classmacro', 'ClassFtoC', 'ClassCtoK', 'ClassReplace', 'ClassTimeshift', 'ClassRH_WtoI', 'Classea', 'Classabs', 'Classrh', 'Classsin', 'Classcos','Classramp', 'Classsquare','Classpulse','Classexp', 'Classpoly', 'Classpolyv', 'Classlog', 'Classpow','Classpowv','Classtime', 'Classjulian', 'Classrandom', 'Classrefwind', 'Classadd', 'Classsub', 'Classmul', 'Classdiv', 'ClassaddV', 'ClasssubV','ClassmulV','ClassdivV', 'Classconst','ClassSim']
mappingFileClass[cpp_file_list[5]] = ['ClassModule']
mappingFileClass[cpp_file_list[6]]
mappingFileClass[cpp_file_list[7]]
mappingFileClass[cpp_file_list[8]]
mappingFileClass[cpp_file_list[9]]
mappingFileClass[cpp_file_list[10]]
mappingFileClass[cpp_file_list[11]]
mappingFileClass[cpp_file_list[12]]
mappingFileClass[cpp_file_list[13]]
mappingFileClass[cpp_file_list[14]]
mappingFileClass[cpp_file_list[15]]
mappingFileClass[cpp_file_list[16]]
mappingFileClass[cpp_file_list[17]]
mappingFileClass[cpp_file_list[18]]
mappingFileClass[cpp_file_list[19]]
mappingFileClass[cpp_file_list[20]]
mappingFileClass[cpp_file_list[21]]
mappingFileClass[cpp_file_list[22]]
mappingFileClass[cpp_file_list[23]]
mappingFileClass[cpp_file_list[24]]
mappingFileClass[cpp_file_list[25]]


def extract_info(ln):
    start = ln.find('::')+2
    end = ln.find('(')
    file.append(fileName)
    className.append(classNameForDef)
    functionName.append(ln[start:end])
    numberOfParameters.append(ln.count(',')+1)
    return


for line in fin:
    line.encode('utf-8')
    locate = line.find(classNameForDef+'::')
    if locate != -1:
        extract_info(line)

df = pd.DataFrame({'file': file, 'className': className, 'functionName': functionName, 'numberOfParameters': numberOfParameters})

df.to_csv('function')

print(functionName)
print(numberOfParameters)


