
def test_strike_a_match():
    ep1 = ['58', '338', '342', '340'] #['F', 'R', 'A', 'N', 'C', 'E']
    ep2 = ['58', '342', '340'] # ['F', 'R', 'E', 'N', 'C', 'H']
    assert compare_execution_paths(ep1, ep2)== 0.6, 'Test Successful'

def method_pairs(execution_path):
    all_pairs = []
    for i in range(len(execution_path) - 1):
        all_pairs.append(execution_path[i: i+2])
        
    return all_pairs
    
def compare_execution_paths(ep1, ep2):
    ep1_pairs = method_pairs(ep1)
    ep2_pairs = method_pairs(ep2)
    union = len(ep1_pairs) + len(ep2_pairs)
    intersection = 0
    
    for i in range(len(ep1_pairs)):
        for j in range(len(ep2_pairs)):
            
            if ep1_pairs[i] == ep2_pairs[j]:
                intersection += 1
                ep2_pairs.pop(j)
                break
                
    return 1 - ( 2 * intersection) / union
    