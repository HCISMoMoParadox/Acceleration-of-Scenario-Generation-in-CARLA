import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mean = list()
    for i in range(1, 26):
        data = np.load('./3_t1-2_0_m_f_l_1_0/cem_iter=%d.npy' % i, allow_pickle=True)
        print(data)
        # a, b, c, d, e, f, g, h, i= data.item()
        a = data.tolist()
        print(a['obj_mean'][0])
        # print(np.extract(data, 6))
        # print(data.item(6))
        mean.append(a['obj_mean'][i-1])
    
    X = np.array(range(1, 26))
    plt.plot(X, mean)
    plt.show()


        # 6 16 18 19