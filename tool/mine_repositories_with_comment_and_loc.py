from PlayingWithAST import *
import os
import random

ROOT = '/Users/avijitbhattacharjee/github/higher_level_abstraction/tool'
def read_csv_file_for_repos():
    with open(ROOT + '/dataset/repoList_python.txt') as f:
        repos = [line.rstrip() for line in f]

    return random.sample(repos, 5)

def count_total_loc(all_files):
    total_loc = 0

    for file in all_files:
        open_file = open(file, 'r')
        total_loc += len(open_file.readline())

    return total_loc

def copy_repo(repo_name):

    url = 'https://github.com/'+repo_name
    os.system('git clone ' + url+ ' ' + ROOT + '/temp/' + repo_name.split('/')[1])
    
    return

def remove_repo(repo):

    os.system('rm -rf '+ ROOT + '/temp/'+ repo.split('/')[1] + '/')

    return

def count_comment_loc_for_repo(location):
    pwa = PlayingWithAST()
    print('Repo: ', location)
    print(count_total_loc(pwa.get_all_py_files(ROOT + '/temp/' + location.split('/')[1]))) 
    print(pwa.get_function_to_comment_ratio(ROOT + '/temp/' + location.split('/')[1]))

    return

if __name__ == "__main__":
    
    repos = read_csv_file_for_repos()
    for repo in repos:
        copy_repo(repo)
        count_comment_loc_for_repo(repo)
        remove_repo(repo)
