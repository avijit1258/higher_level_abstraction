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
                'EntryForm.cpp','Examples.cpp','Export.cpp','GlobalCommon.cpp','GlobalDll.cpp','Hype_CRHM.cpp','Hype_lake.cpp', \
                'Hype_river.cpp','Hype_routines.cpp','Log.cpp','MacroUnit.cpp','NewModules.cpp','Numerical.cpp','Para.cpp','report.cpp','UpdateForm.cpp']

mappingFileClass = {}

mappingFileClass[cpp_file_list[0]] = ['TAboutBox']
mappingFileClass[cpp_file_list[1]] = ['TFormAKA']
mappingFileClass[cpp_file_list[2]] = ['Plot', 'TAnalysis']
mappingFileClass[cpp_file_list[3]] = ['TBldForm']
#3975 number line of classCRHM.cpp is considered
mappingFileClass[cpp_file_list[4]] = ['ClassVar', 'ClassPar','ClassData','Classmacro', 'ClassFtoC', 'ClassCtoK', 'ClassReplace', 'ClassTimeshift', 'ClassRH_WtoI', 'Classea', 'Classabs', 'Classrh', 'Classsin', 'Classcos','Classramp', 'Classsquare','Classpulse','Classexp', 'Classpoly', 'Classpolyv', 'Classlog', 'Classpow','Classpowv','Classtime', 'Classjulian', 'Classrandom', 'Classrefwind', 'Classadd', 'Classsub', 'Classmul', 'Classdiv', 'ClassaddV', 'ClasssubV','ClassmulV','ClassdivV', 'Classconst','ClassSim']
mappingFileClass[cpp_file_list[5]] = ['ClassModule']
mappingFileClass[cpp_file_list[6]] = [] # Common.cpp has no mentionable class
mappingFileClass[cpp_file_list[7]] = ['VarCHRM']
mappingFileClass[cpp_file_list[8]] = ['TMain']
mappingFileClass[cpp_file_list[9]] = ['TMain']
mappingFileClass[cpp_file_list[10]] = ['TFormEntry']
mappingFileClass[cpp_file_list[11]] = ['TLibForm']
mappingFileClass[cpp_file_list[12]] = ['TFileOutput']
mappingFileClass[cpp_file_list[13]] = []
mappingFileClass[cpp_file_list[14]] = []
mappingFileClass[cpp_file_list[15]] = ['ClassWQ_Soil', 'ClassWQ_Netroute', 'ClassWQ_ion', 'ClassWQ_pbsm']
mappingFileClass[cpp_file_list[16]] = ['ClassWQ_Lake']
mappingFileClass[cpp_file_list[17]] = ['ClassWQ_River', 'ClassWQ_REWroute']
mappingFileClass[cpp_file_list[18]] = ['ClassWQ_Hype']
mappingFileClass[cpp_file_list[19]] = ['TLogForm']
mappingFileClass[cpp_file_list[20]] = ['ClassMacro', 'Defdeclparam', 'Defdecldiag', 'Defdeclstatvar', 'Defdecllocal', 'Defdeclgetvar', 'Defdeclputvar', 'Defdeclputparam', 'Defdeclreadobs', 'Defdeclobsfunc']
mappingFileClass[cpp_file_list[21]] = ['Classshared', 'ClassNOP', 'Classbasin', 'Classglobal', 'Classobs', 'Classintcp', 'Classpbsm', 'ClassSoilDS','Classalbedoobs2','Classwinter_meltflag','Class_z_s_rho','Classtsurface','Classqdrift','Classqmelt', 'Classquinton','ClassICEflow'] #start from NewModules.cpp
mappingFileClass[cpp_file_list[22]] = ['Poly','Fourier','Power','Expo', 'Log', 'MLinReg','LeastSquares']#Numerical.cpp
mappingFileClass[cpp_file_list[23]] = ['TParameter']#Para.cpp
mappingFileClass[cpp_file_list[24]] = ['TRprt']#report.cpp
mappingFileClass[cpp_file_list[25]] = ['TPlotControl']#report.cpp



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


