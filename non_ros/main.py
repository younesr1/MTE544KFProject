#!/usr/bin/python3
from sensor_msgs.msg import Imu, JointState 
import pickle

def read_data():
    f = open('jnt.pkl','rb')
    jnt = pickle.load(f)
    f.close()
    f = open('imu.pkl','rb')
    imu = pickle.load(f)
    f.close()
    return imu,jnt

def main():
    imu,jnt = read_data()
    print(len(imu[::6]),len(jnt))

if __name__ == '__main__':
    main()
