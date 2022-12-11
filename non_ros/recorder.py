import rclpy
from rclpy.node import Node
import pickle
import signal

from std_msgs.msg import String
from sensor_msgs.msg import Imu, JointState
import sys



class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        print('created subscriber')
        self.imu_msg = []
        self.jnt_msg = []
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            rclpy.qos.qos_profile_sensor_data)
        self.jnt_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.jnt_cb,
            10)

    def save_data(self):
        with open('imu.pkl', 'wb') as f:
            pickle.dump(self.imu_msg, f)
        with open('jnt.pkl', 'wb') as f:
            pickle.dump(self.jnt_msg, f)

    def imu_cb(self, msg):
        assert type(msg) == Imu
        self.imu_msg.append(msg)

    def jnt_cb(self, msg):
        assert type(msg) == JointState
        self.jnt_msg.append(msg)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()
    def signal_handler(sig,frame):
        minimal_subscriber.save_data()
        print('terminating')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
