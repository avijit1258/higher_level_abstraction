

fileName = 'CRHMmain.cpp'
infname = 'C:/Users/Saikat Mondal/PycharmProjects/HigerLevelAbstraction/instrumenting source code/original_sources/'+fileName

file = []
className = []
FunctionName = []
NumberOfParameters = []


fin = open(infname)

def extract_info(ln):
    start = ln.find('::')+2
    end = ln.find('(')
    print(ln[start:end])
    return


for line in fin:
    line.encode('utf-8')
    locate = line.find('TMain::')
    if locate != -1:
        print(locate)
        print(line)
        extract_info(line)



