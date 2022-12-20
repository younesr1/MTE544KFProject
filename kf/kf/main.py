import rclpy
import signal
import sys
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformException                       
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class KF(Node):
    def __init__(self):
        super().__init__('kf')
        self.imu_msg = None
        self.odom_msg = None
        # we get imu at 200Hz but odom at 30Hz
        # we decide to run KF at 20 Hz to be conservative
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            rclpy.qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            10)
        # run the KF at 20 Hz bc Odometry comes at 30 Hz
        self.T = 1/20
        self.kf_pub = self.create_publisher(Pose, '/kf/pose', int(1/self.T))
        self.timer = self.create_timer(self.T, self.update)
        # the state x is defined as x:=[x,y,x_dot,y_dot,theta]
        # the 'input' u is defined as u:=[ax,ay,w] (from IMU)
        # the measurement y is defined as y:=[x,y,theta] (from odom)
        #x[+1]=Ax+Bu+e where e~N(0,Q)
        #y=Cx+delta where delta~N(0,R)
        self.A = np.matrix([[1,0,self.T,0,0],[0,1,0,self.T,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
        assert self.A.shape == (5,5)
        self.B = np.matrix([[(self.T*self.T/2),0,0],[0,(self.T*self.T/2),0],[self.T,0,0],[0,self.T,0],[0,0,self.T]])
        assert self.B.shape == (5,3)
        self.C = np.matrix([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1]])
        assert self.C.shape == (3,5)
        # This is gotten from the IMU msg
        Qa = np.diag([0.000289,0.000289,4e-8])
        assert Qa.shape == (3,3)
        self.Q = self.B@Qa@self.B.transpose()
        assert self.Q.shape == (5,5)
        # this convar is gotten from odom msg
        self.R = np.diag([1e-5,1e-5,0.001])
        assert self.R.shape == (3,3)
        self.x_hat = np.zeros((5,1))
        assert self.x_hat.shape == (5,1)
        self.P = np.eye(5)*1e-5
        assert self.P.shape == (5,5)
        # these variables store values for calculating MSE
        self.x_observed = []
        self.y_observed = []
        self.theta_observed = []
        self.x_gt = []
        self.y_gt= []
        self.theta_gt = []
        # this is the tf listner for the ground truth
        # we only use the ground truth for MSE
        self.tf_buffer = Buffer()                            
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def imu_cb(self, msg):
        assert type(msg) == Imu
        self.imu_msg = msg

    def odom_cb(self, msg):
        assert type(msg) == Odometry
        self.odom_msg = msg

    # in our case u is the IMU reading
    # in our formulation, u is a 3x1 vector as [ax,ay,w]
    # this returns u
    def get_u(self):
        # planar robot only turns in z
        #assert math.isclose(self.imu_msg.angular_velocity.x,0,abs_tol=1e-2)
        #assert math.isclose(self.imu_msg.angular_velocity.y,0,abs_tol=1e-2)
        ax = self.imu_msg.linear_acceleration.x
        ay = self.imu_msg.linear_acceleration.y
        az = self.imu_msg.linear_acceleration.z
        a = np.array([ax,ay,az])
        assert a.shape == (3,)
        quat = self.imu_msg.orientation
        a = R.from_quat([quat.x,quat.y,quat.z,quat.w]).apply(a)
        assert a.shape == (3,)
        #assert math.isclose(a[2],self.imu_msg.linear_acceleration.z,abs_tol=1e-3)
        # currently, a=[ax,ay,az], but u = [ax,ay,w]
        # lets return u
        u = np.vstack((np.reshape(a[:-1],(2,1)),self.imu_msg.angular_velocity.z))
        assert u.shape == (3,1)
        return u

    # y is the sensor value. our sensor is odom which reports x,y,theta
    # this function converts this ros msg to a 3x1 matrix
    def get_y(self):
        pose = self.odom_msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        quat = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
        theta = R.from_quat(quat).as_euler('xyz',degrees=False)
        #assert math.isclose(theta[0],0,abs_tol=1e-2)
        #assert math.isclose(theta[1],0,abs_tol=1e-2)
        assert np.radians(-30)<theta[2]<np.radians(30)
        ret = np.matrix([x,y,theta[2]]).transpose()
        assert ret.shape == (3,1)
        return ret
        

    def update(self):
        if not self.imu_msg or not self.odom_msg:
            self.get_logger().info('Waiting for IMU and JointState messages to arrive')
            return
        u = self.get_u()
        assert u.shape == (3,1)
        y = self.get_y()
        assert y.shape == (3,1)
        # prediction step
        x_hat_pred = self.A@self.x_hat+self.B@u
        assert x_hat_pred.shape == (5,1)
        p_pred = self.A@self.P@self.A.transpose()+self.Q
        assert p_pred.shape == (5,5)
        # update step
        K = p_pred@self.C.transpose()@(np.linalg.inv(self.C@p_pred@self.C.transpose()+self.R))
        assert K.shape == (5,3)
        self.x_hat = x_hat_pred+K@(y-self.C@x_hat_pred)
        assert self.x_hat.shape == (5,1)
        self.P = (np.identity(5)-K@self.C)@p_pred
        assert self.P.shape == (5,5)
        # publish the pose kf pose for convenience
        msg = Pose()
        msg.position.x = self.x_hat[0,0]
        msg.position.y = self.x_hat[1,0]
        q = msg.orientation
        q.x,q.y,q.z,q.w = R.from_euler('z',self.x_hat[4,0],degrees=False).as_quat()
        self.kf_pub.publish(msg)
        # get the groudn truth
        try:
            t = self.tf_buffer.lookup_transform('odom','base_link',rclpy.time.Time())
            self.x_gt.append(t.transform.translation.x)
            self.y_gt.append(t.transform.translation.y)
            rot = t.transform.rotation
            quat = [rot.x,rot.y,rot.z,rot.w]
            theta = R.from_quat(quat).as_euler('xyz')[-1]
            self.theta_gt.append(theta)
            # collect the arrays
            self.x_observed.append(self.x_hat[0,0])
            self.y_observed.append(self.x_hat[1,0])
            self.theta_observed.append(self.x_hat[4,0])
            # compute the mse values
            assert len(self.x_observed) == len(self.x_gt)
            assert len(self.y_observed) == len(self.y_gt)
            assert len(self.theta_observed) == len(self.theta_gt)
            x_mse = np.square(np.array(self.x_observed)-np.array(self.x_gt)).mean()
            y_mse = np.square(np.array(self.y_observed)-np.array(self.y_gt)).mean()
            theta_mse = np.square(np.array(self.theta_observed)-np.array(self.theta_gt)).mean()
            self.get_logger().info(f'MSE Stats: x: {x_mse}, y: {y_mse}, theta: {theta_mse}')
        except TransformException as e:
            self.get_logger().fatal(f'Could not get the transform. Terminating')

        
        self.imu_msg = None
        self.odom_msg = None

def main(args=None):
    rclpy.init(args=args)
    kf = KF()
    rclpy.spin(kf)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    kf.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
