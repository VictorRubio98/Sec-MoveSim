from evaluations import *
from utils import *

def distance(gps1,gps2):
    x1,y1 = gps1
    x2,y2 = gps2
    return np.sqrt((x1-x2)**2+(y1-y2)**2 )

    
    

def gen_matrix(data='geolife'):
    train_data = read_data_from_file(f'../data/{data}/real.data')
    gps = get_gps(f'../data/{data}/gps')
    if data=='mobile':
        max_locs = 8606
    elif data == 'geolife':
        max_locs = 23768
    elif data == 'porto':
        max_locs = len(gps[0])
    else:
        max_locs = 0

    print(f'Maximum locations on GPS file: {max_locs}', flush=True)

    reg1 = np.zeros([max_locs,max_locs])
    for i in range(len(train_data)):
        line = train_data[i]
        for j in range(len(line)-1):
            reg1[line[j],line[j+1]] +=1
    reg2 = np.zeros([max_locs,max_locs])
    for i in range(max_locs):
        for j in range(max_locs):
            if i!=j:
                reg2[i,j] = distance((gps[0][i],gps[1][i]),(gps[0][j],gps[1][j]))
    

    np.save(f'../data/{data}/M1.npy',reg1)
    np.save(f'../data/{data}/M2.npy',reg2)

    print('Matrix Generation Finished', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data',default='geolife', type=str)
    opt = parser.parse_args()
    gen_matrix(opt.data)

    




    
