from tools import remove_comments
from tools import fn_start_inspect
from tools import fn_inject
from tools import fn_inject_inspect

# main file
# |
# V

#remove_comments('CRHMmain.cpp', 'CRHMmain_nocom.cpp')

# file without comments
# |
# V

#fn_start_inspect('CRHMmain_nocom.cpp')
# manually save the file by modifying lines if necessary

# inspection in no comments file and modified version (a new one or overwritten one)
# |
# V

cpp_file_list = ['About.cpp','AKAform.cpp','Analy.cpp','Bld.cpp','ClassCRHM.cpp','ClassModule.cpp','Common.cpp','CRHM_parse.cpp','CRHMmain.cpp' \
                'EntryForm.cpp','Examples.cpp','Export.cpp','Export.cpp','GlobalCommon.cpp','GlobalDll.cpp','Hype_CRHM.cpp','Hype_lake.cpp', \
                'Hype_river.cpp','Hype_routines.cpp','Log.cpp','MacroUnit.cpp','NewModules.cpp','Numerical.cpp','Para.cpp','report.cpp','UpdateForm.cpp']

for i in cpp_file_list:
    fn_inject(i, 'I_'+i)
    print(i)
#CRHMmain
#Hype_CRHM
#Hype_lake
#fn_inject('Hype_lake.cpp', 'I_Hype_lake.cpp')

# code injected file
# |
# v

#fn_inject_inspect('CRHMmain_injected.cpp')

# soft inspection of injected file
