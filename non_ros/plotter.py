import matplotlib.pyplot as plt
import numpy as np
import pickle



def main():
    f = open('data.pkl','rb')
    x,y,t,xgt,ygt,tgt = pickle.load(f)
    assert len(x) == len(y) == len(t) == len(xgt) == len(ygt) == len(tgt)
    '''fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x,y,c='b',s=1,label='KF Algorithm')
    ax1.scatter(xgt,ygt,c='r',s=1,label='Ground Truth')
    plt.legend(loc='upper left')
    plt.title('Algorithm to Ground Truth Comparison')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.show()'''
    x_mse = [np.square(np.array(x[:i+1])-np.array(xgt[:i+1])).mean() for i in range(len(x))]
    y_mse = [np.square(np.array(y[:i+1])-np.array(ygt[:i+1])).mean() for i in range(len(x))]
    t_mse = [np.square(np.array(t[:i+1])-np.array(tgt[:i+1])).mean() for i in range(len(x))]
    plt.plot(x_mse,'r',label='X MSE')
    plt.plot(y_mse,'b',label='Y MSE')
    plt.plot(t_mse,'g',label='Theta MSE')
    plt.ylabel("MSE")
    plt.xlabel("Index")
    plt.title('X, Y, Theta MSE')
    plt.legend()
    plt.show()
    f.close()

main()
