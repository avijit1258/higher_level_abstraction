def remove_comments(infname, outfname):

    fin = open(infname)
    fout = open(outfname, 'w')

    for line in fin:
        if '//' in line:
            fout.write( line[ : line.index('//') ] + '\n' )
        else:
            fout.write(line)

def fn_start_inspect(infname):
    fin = open(infname)

    prev = ''
    for line in fin:
        if line == '{\n':
            print(prev)


        prev = line

def fn_inject(infname, outfname):

    fin = open(infname)
    fout = open(outfname, 'w')

    save = ''
    prev = ''
    #fin.encode('utf-8')
    for line in fin:
        line.encode('utf-8')
        print(line)

        if len(line.strip()) == 0:
            pass

        elif line.strip() == '{' and line[0] == '{':
            #injection_msg = 'freopen(\"injection.xml", "a", stdout); printf("<' + ''.join(prev.strip().split()) + ' @@@ ' + 'CRHMmain.cpp_nocom' + '>" \n); fclose(stdout);'
            injection_msg = 'Global::CRHMInstLogs->Add("<' + ''.join(prev.strip().split()) + ' @@@ ' + infname+'_nocom' + '>\\n\");'
            fout.write('{\n' + injection_msg + '\n')
            save = prev

        elif line.strip() == '}' and line[0] == '}':
            #injection_msg = 'freopen(\"injection.xml", "a", stdout); printf("</' + ''.join(save.strip().split()) +  ' @@@ ' +  'CRHMmain.cpp_nocom' +'>\" \n); fclose(stdout);'

            injection_msg = 'Global::CRHMInstLogs->Add("</' + ''.join(save.strip().split()) +  ' @@@ ' +infname+  '_nocom' +'>\\n\");'
            fout.write(injection_msg + '\n' + '}\n')

        elif line.strip().split()[0][0:6] == 'return':
            #injection_msg = '{' + 'freopen(\"injection.xml", "a", stdout); printf("</' + ''.join(save.strip().split()) + ' @@@ ' + 'CRHMmain.cpp_nocom' + '>" \n); fclose(stdout);' + line.strip() + '}'

            injection_msg = '{' + 'Global::CRHMInstLogs->Add("</' + ''.join(save.strip().split()) + ' @@@ ' +infname+ '_nocom' + '>\\n\");' + line.strip() + '}'
            fout.write(injection_msg + '\n')

        else:
            fout.write(line)

        prev = line


def fn_inject_inspect(infname):
    fin = open(infname)

    for line in fin:
        if 'start_@ ' in line or 'end_@ ' in line:
            print(line)

