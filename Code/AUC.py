def AUC(probs_list):
    # very simplistic version, maybe not 100% accurate -> only approximation
    ''' Calculates the area under the receiver operant characteristic curve
        given an array with labels '1' and '0' (strings) and predicted
        probabilities
    '''
    probs_list = probs_list[probs_list[:,1].argsort()]
    
    cumulative = 0;
    y = 0
    num_pos = probs_list[probs_list==('1')].size
    num_neg = len(probs_list) - num_pos
    delta_y = 1 / num_pos
    delta_x = 1 / num_neg
    
    for x in range(len(probs_list)):
        if (probs_list[-1-x][0]=='1'):
            y = y + delta_y
        else:
            cumulative = cumulative + (y * delta_x)
    
    return cumulative  