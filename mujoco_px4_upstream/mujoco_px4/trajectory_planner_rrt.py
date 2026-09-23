import rclpy
from rclpy.node import Node
from rtree import index
import numpy as np
from numpy import linalg as LA
from scipy import optimize
from collections import namedtuple
from px4_msgs.msg import TrajectorySetpoint
from math import ceil
import time
"""
    This node generates the trajectory to be tracked by the geometeric controller
    Publisher:
        mujoco_px4/trajectory_setpoint
    Subscriber:
        None
        start, goal, and map are known
"""
class RRTNode:
    def __init__(self,coords):
        self.p = np.array(coords)
        self.parent = None
        self.cost = np.inf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.p[i]

    def __repr__(self):
        return 'RRTNode({}, {})'.format(self.p,self.cost)

#Rtree to store Nodes
class Rtree:
  def __init__(self,dim):
    self.dim = dim
    self.node_list = []
    self.idx = self.get_tree(dim)
    self.len = 0

  @staticmethod
  def get_tree(dim):
    '''Initialise the tree'''
    p = index.Property()
    p.dimension = dim
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    return index.Index(properties=p)

  def add(self,new_node):
    '''add nodes to tree'''
    self.node_list.append(new_node)
    self.idx.insert(self.len,(*new_node.p,))
    self.len += 1

  def k_nearest(self,node,k):
    '''Returns k-nearest nodes to the given node'''
    near_ids = self.idx.nearest((*node.p,),k)
    for i in near_ids:
      yield self.node_list[i]

  def nearest(self,node):
    '''Returns nearest node to the given node'''
    near_ids = self.idx.nearest((*node.p,),1)
    id = list(near_ids)[0]
    return self.node_list[id]

  def all(self):
    return self.node_list
  

class Map:
  def __init__(self,obstacle_list,bounds,path_resolution = 0.5,dim = 3):
    self.dim = dim
    self.idx = self.get_tree(obstacle_list,dim)
    self.len = len(obstacle_list)
    self.path_res = path_resolution
    self.obstacles = obstacle_list
    self.bounds = bounds

  @staticmethod
  def get_tree(obstacle_list,dim):
    '''initialise map with given obstacle_list'''
    p = index.Property()
    p.dimension = dim
    ls = [(i,(*obj,),None) for i, obj in enumerate(obstacle_list)]
    return index.Index(ls, properties=p)

  def add(self,obstacle):
    '''add new obstacle'''
    self.idx.insert(self.len,obstacle)
    self.obstacles.append(obstacle)
    self.len += 1

  def collision(self,start,end):
    '''find if the ray between start and end collides with obstacles'''
    dist = np.linalg.norm(start-end)
    n = int(dist/self.path_res)
    points = np.linspace(start,end,n)
    for p in points:
      if self.idx.count((*p,)) != 0 :
          return True
    return False

  def inbounds(self,p):
      '''Check if p lies inside map bounds'''
      lower,upper = self.bounds
      return (lower <= p).all() and (p <= upper).all()
  

class RRT:

    def __init__(self, start, goal, Map,
                 max_extend_length = 3.0,
                 path_resolution = 0.5,
                 goal_sample_rate = 0.05,
                 max_iter = 100 ):
        self.start = RRTNode(start)
        self.goal = RRTNode(goal)
        self.max_extend_length = max_extend_length
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.dim = start.shape[0]
        self.tree = Rtree(self.dim)
        self.map = Map

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.tree.add(self.start)
        for i in range(self.max_iter):
            #Generate a random node (rnd_node)
            rnd_node = self.get_random_node()
            #Get nearest node (nearest_node)
            nearest_node = self.tree.nearest(rnd_node)
            #Get new node (new_node) by connecting
            new_node = self.steer(nearest_node,rnd_node)
            #If the path between new_node and the nearest node is not in collision
            if not self.map.collision(nearest_node.p,new_node.p):
              self.tree.add(new_node)
              # If the new_node is very close to the goal, connect it
              # directly to the goal and return the final path
              if self.dist(new_node,self.goal) <= self.max_extend_length:
                  if not self.map.collision(new_node.p,self.goal.p):
                      self.goal.parent = new_node
                      return self.final_path()
        # cannot find path
        return None

    @staticmethod
    def dist(from_node, to_node):
        #euler distance
        return np.linalg.norm(from_node.p - to_node.p)

    def steer(self,from_node, to_node):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        dist = self.dist(from_node, to_node)
        #Rescale the path to the maximum extend_length
        if dist > self.max_extend_length:
            diff = from_node.p - to_node.p
            to_node.p  = from_node.p - diff/dist * self.max_extend_length
        to_node.parent = from_node
        return to_node

    def sample(self):
        # Sample random point inside boundaries
        lower,upper = self.map.bounds
        return lower + np.random.rand(self.dim)*(upper - lower)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            rnd = self.sample()
        else:
            rnd = self.goal.p
        return RRTNode(rnd)

    def final_path(self):
        """Compute the final path from the goal node to the start node"""
        path = []
        node = self.goal
        if (node.p == node.parent.p).all(): node = node.parent
        while node.parent:
          path.append(node.p)
          node = node.parent
        path.append(self.start.p)
        return np.array(path[::-1]) # reverse the np array from start->goal

    # def draw_graph(self,ax):
    #     '''plot the whole graph'''
    #     for node in self.tree.all():
    #         if node.parent:
    #             xy = np.c_[node.p,node.parent.p]
    #             ax.plot(*xy, "-g",zorder = 5)

    # def draw_path(self,ax,path, color=(0.9, 0.2, 0.5, 0.8)):
    #     '''draw the path if available'''
    #     if path is None:
    #         print("path not available")
    #     else:
    #         ax.plot(*np.array(path).T, '-', color = color, zorder = 5)

    # def draw_scene(self,path = None,ax = None):
    #     '''draw the whole scene'''
    #     if ax is None:
    #         fig = plt.figure()
    #         if self.dim == 3:
    #             ax = Axes3D.Axes3D(fig)
    #         elif self.dim == 2:
    #             ax = plt.axes()
    #         else:
    #             print('cannot plot for current dimensions')
    #             return
    #     self.draw_graph(ax)
    #     self.draw_path(ax,path)
    #     self.map.plotobs(ax)
    #     plt.show()

def SampleUnitNBall(dim = 3,num = 1):
    '''
    uniformly sample a N-dimensional unit UnitBall
    Reference:
      Efficiently sampling vectors and coordinates from the n-sphere and n-ball
      http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    Input:
        num - no. of samples
        dim - dimensions

    Output:
        uniformly sampled points within N-dimensional unit ball
    '''
    #Sample on a unit N+1 sphere
    u = np.random.normal(0, 1, (num, dim + 2))
    norm = LA.norm(u, axis = -1,keepdims = True)
    u = u/norm
    #The first N coordinates are uniform in a unit N ball
    if num == 1: return u[0,:dim]
    return u[:,:dim]


class EllipsoidSampler:
    '''
    uniformly sample within a N-dimensional Ellipsoid
    Reference:
      Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct Sampling
      of an Admissible Ellipsoidal Heuristic https://arxiv.org/pdf/1404.2334.pdf
    '''
    def __init__(self,center,axes = [],rot = []):
        '''
        Input:
            center -  centre of the N-dimensional ellipsoid in the N-dimensional
            axes -  axes length across each dimension in ellipsoid frame
            rot - rotation matrix from ellipsoid frame to world frame
        Output:
            uniformly sampled points within the hyperellipsoid
        '''
        self.dim = center.shape[0]
        self.center = center
        self.rot = rot
        if len(rot) == 0: self.rot = np.eye(self.dim)
        if len(axes) == 0: axes = [1]*self.dim
        self.L = np.diag(axes)

    def sample(self,num = 1):
        xball = SampleUnitNBall(self.dim,num)
        #Transform points in UnitBall to ellipsoid
        xellip = (self.rot@self.L@xball.T).T + self.center
        return xellip



class InformedSampler:
    '''
    uniformly sample within a N-dimensional Prolate-Hyperspheroid for informed RRT*
    with goal and start as focal points
    Reference:
      Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct Sampling
      of an Admissible Ellipsoidal Heuristic https://arxiv.org/pdf/1404.2334.pdf
    '''
    def __init__(self, goal, start):
        self.dim = goal.shape[0]
        self.cmin = LA.norm(goal - start)
        center = (goal + start)/2
        #rotation matrix from ellipsoid frame to world frame
        C = self.RotationToWorldFrame(goal, start)
        #initialise EllipsoidSampler
        self.ellipsampler = EllipsoidSampler(center,rot = C)

    def sample(self, cmax, num = 1):
        '''
        Input:
            cmax - current best cost
            num - no. of samples
        Output:
            uniformly sampled point within informed region
        '''
        #Hyperspheroid axes lengths
        r1 = cmax/2
        ri = np.sqrt(cmax**2 - self.cmin**2)/2
        axes = [r1]+ [ri]*(self.dim - 1)
        self.ellipsampler.L = np.diag(axes)
        #return sampled point
        return self.ellipsampler.sample(num)

    def RotationToWorldFrame(self, goal, start):
        '''
        Given two focal points goal and start in N-Dimensions
        Returns rotation matrix from the ellipsoid frame to the world frame
        '''
        #Transverse axis of the ellipsoid in the world frame
        E1 = (goal - start) / self.cmin
        #first basis vector of the world frame [1,0,0,...]
        W1 = [1]+[0]*(self.dim - 1)
        #outer product of E1 and W1
        M = np.outer(E1,W1)
        #SVD decomposition od outer product
        U, S, V = LA.svd(M)
        #Calculate the middle diagonal matrix
        middleM = np.eye(self.dim)
        middleM[-1,-1] = LA.det(U)*LA.det(V)
        #calculate the rotation matrix
        C = U@middleM@V.T
        return C


class RRTStar(RRT):

    def __init__(self, start, goal, Map,
                 max_extend_length = 3.0,
                 path_resolution = 0.5,
                 goal_sample_rate = 0.05,
                 max_iter = 200 ):
        super().__init__(start, goal, Map, max_extend_length,
                         path_resolution, goal_sample_rate, max_iter)
        self.final_nodes = []
        self.Informedsampler = InformedSampler(goal, start)

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.start.cost = 0
        self.tree.add(self.start)
        for i in range(self.max_iter):
            #Generate a random node (rnd_node)
            rnd = self.get_random_node()
            # Get nearest node
            nearest_node = self.tree.nearest(rnd)
            # Get new node by connecting rnd_node and nearest_node
            new_node = self.steer(nearest_node, rnd)
            # If path between new_node and nearest node is not in collision
            if not self.map.collision(nearest_node.p,new_node.p):
              #add the node to tree
              self.add(new_node)
        #Return path if it exists
        if not self.goal.parent: path = None
        else: path = self.final_path()
        return path, self.goal.cost

    def add(self,new_node):
        near_nodes = self.near_nodes(new_node)
        # Connect the new node to the best parent in near_inds
        self.choose_parent(new_node,near_nodes)
        #add the new_node to tree
        self.tree.add(new_node)
        # Rewire the nodes in the proximity of new_node if it improves their costs
        self.rewire(new_node,near_nodes)
        #check if it is in close proximity to the goal
        if self.dist(new_node,self.goal) <= self.max_extend_length:
          # Connection between node and goal needs to be collision free
          if not self.map.collision(self.goal.p,new_node.p):
            #add to final nodes if in goal region
            self.final_nodes.append(new_node)
        #set best final node and min_cost
        self.choose_parent(self.goal,self.final_nodes)

    def choose_parent(self, node, parents):
        """Set node.parent to the lowest resulting cost parent in parents and
           node.cost to the corresponding minimal cost
        """
        # Go through all near nodes and evaluate them as potential parent nodes
        for parent in parents:
          #checking whether a connection would result in a collision
          if not self.map.collision(node.p,parent.p):
            #evaluating the cost of the new_node if it had that near node as a parent
            cost = self.new_cost(parent, node)
            #picking the parent resulting in the lowest cost and updating the cost of the new_node to the minimum cost.
            if cost < node.cost:
              node.parent = parent
              node.cost = cost

    def rewire(self, new_node, near_nodes):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        #Go through all near nodes and check whether rewiring them to the new_node is useful
        for node in near_nodes:
          self.choose_parent(node,[new_node])
        self.propagate_cost_to_leaves(new_node)

    def near_nodes(self, node):
        """Find the nodes in close proximity to given node"""
        nnode = self.tree.len + 1
        r = ceil(5.5*np.log(nnode))
        return self.tree.k_nearest(node,r)

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        return from_node.cost + self.dist(from_node, to_node)

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.tree.all():
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def sample(self):
        """Sample random node inside the informed region"""
        lower,upper = self.map.bounds
        if self.goal.parent:
          rnd = np.inf
          #sample until rnd is inside bounds of the map
          while not self.map.inbounds(rnd):
              # Sample random point inside ellipsoid
              rnd = self.Informedsampler.sample(self.goal.cost)
        else:
          # Sample random point inside boundaries
          rnd = lower + np.random.rand(self.dim)*(upper - lower)
        return rnd
    
DesiredState = namedtuple('DesiredState', 'pos vel acc jerk yaw yawdot')

def polyder(t, k = 0, order = 10):
    if k == 'all':
        terms = np.array([polyder(t,k,order) for k in range(1,5)])
    else:
        terms = np.zeros(order)
        coeffs = np.polyder([1]*order,k)[::-1]
        pows = t**np.arange(0,order-k,1)
        terms[k:] = coeffs*pows
    return terms

def Hessian(T,order = 10,opt = 4):
    n = len(T)
    Q = np.zeros((order*n,order*n))
    for k in range(n):
        m = np.arange(0,opt,1)
        for i in range(order):
            for j in range(order):
                if i >= opt and j >= opt:
                    pow = i+j-2*opt+1
                    Q[order*k+i,order*k+j] = 2*np.prod((i-m)*(j-m))*T[k]**pow/pow
    return Q
    
class trajGenerator:
    def __init__(self,waypoints,max_vel = 1.0 ,gamma = 100):
        self.waypoints = waypoints
        self.max_vel = max_vel
        self.gamma = gamma
        self.order = 10
        len,dim = waypoints.shape
        self.dim = dim
        self.len = len
        self.TS = np.zeros(self.len)
        self.optimize()
        self.yaw = 0
        self.heading = np.zeros(2)

    def get_cost(self,T):
        coeffs,cost = self.MinimizeSnap(T)
        cost = cost + self.gamma*np.sum(T)
        return cost

    def optimize(self):
        diff = self.waypoints[0:-1] - self.waypoints[1:]
        Tmin = LA.norm(diff,axis = -1)/self.max_vel
        T = optimize.minimize(self.get_cost,Tmin, method="COBYLA",constraints= ({'type': 'ineq', 'fun': lambda T: T-Tmin}))['x']

        self.TS[1:] = np.cumsum(T)
        self.coeffs, self.cost = self.MinimizeSnap(T)


    def MinimizeSnap(self,T):
        unkns = 4*(self.len - 2)

        Q = Hessian(T) #
        A,B = self.get_constraints(T)

        invA = LA.inv(A)

        if unkns != 0:
            R = invA.T@Q@invA

            Rfp = R[:-unkns,-unkns:]
            Rpp = R[-unkns:,-unkns:]

            B[-unkns:,] = -LA.inv(Rpp)@Rfp.T@B[:-unkns,]

        P = invA@B
        cost = np.trace(P.T@Q@P)

        return P, cost

    def get_constraints(self,T):
        n = self.len - 1
        o = self.order

        A = np.zeros((self.order*n, self.order*n))
        B = np.zeros((self.order*n, self.dim))

        B[:n,:] = self.waypoints[ :-1, : ]
        B[n:2*n,:] = self.waypoints[1: , : ]

        #waypoints contraints
        for i in range(n):
            A[i, o*i : o*(i+1)] = polyder(0)
            A[i + n, o*i : o*(i+1)] = polyder(T[i])

        #continuity contraints
        for i in range(n-1):
            A[2*n + 4*i: 2*n + 4*(i+1), o*i : o*(i+1)] = -polyder(T[i],'all')
            A[2*n + 4*i: 2*n + 4*(i+1), o*(i+1) : o*(i+2)] = polyder(0,'all')

        #start and end at rest
        A[6*n - 4 : 6*n, : o] = polyder(0,'all')
        A[6*n : 6*n + 4, -o : ] = polyder(T[-1],'all')

        #free variables
        for i in range(1,n):
            A[6*n + 4*i : 6*n + 4*(i+1), o*i : o*(i+1)] = polyder(0,'all')

        return A,B

    def get_des_state(self,t):

        if t > self.TS[-1]: t = self.TS[-1] - 0.001

        i = np.where(t >= self.TS)[0][-1]

        t = t - self.TS[i]
        coeff = (self.coeffs.T)[:,self.order*i:self.order*(i+1)]

        pos  = coeff@polyder(t)
        vel  = coeff@polyder(t,1)
        accl = coeff@polyder(t,2)
        jerk = coeff@polyder(t,3)

        #set yaw in the direction of velocity
        yaw, yawdot = self.get_yaw(vel[:2])

        return DesiredState(pos, vel, accl, jerk, yaw, yawdot)

    def get_yaw(self,vel):
        curr_heading = vel/LA.norm(vel)
        prev_heading = self.heading
        cosine = max(-1,min(np.dot(prev_heading, curr_heading),1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading,curr_heading)
        self.yaw += np.sign(norm_v)*dyaw

        if self.yaw > np.pi: self.yaw -= 2*np.pi
        if self.yaw < -np.pi: self.yaw += 2*np.pi

        self.heading = curr_heading
        yawdot = max(-30,min(dyaw/0.005,30))
        return self.yaw,yawdot


class TrajectoryPlannerRRT(Node):

    def __init__(self):
        super().__init__("px4_mujoco_trajectory_planner")

        #initialize the RRT*
        np.random.seed(5)

        # 3D boxes   lx, ly, lz, hx, hy, hz
        self.obstacles = [[-1, 5, -20, 1, 60, 50],
                    [19, -30, -20, 21, 25, 50],
                    [-20, 60, -20, 40, 100, 50],
                    [-20, 0.0, -20, 40, -100, 50],
                    [-10, -10, 0.0, 40, 70, -70]
                    ]


        # limits on map dimensions
        self.bounds = np.array([-20,60])
        # create map with obstacles
        self.mapobs = Map(self.obstacles, self.bounds, dim = 3)

        #plan a path from start to goal
        start = np.array([-10, 20, 10])
        goal = np.array([40, 10, 10])


        rrt = RRTStar(start = start, goal = goal,
                    Map = self.mapobs, max_iter = 2000,
                    goal_sample_rate = 0.1)

        waypoints, min_cost = rrt.plan()


        # scale the waypoints to real dimensions
        waypoints = 0.1*waypoints

        #Generate trajectory through waypoints
        self.traj = trajGenerator(waypoints, max_vel = 1.0, gamma = 1e6)

        #initialise simulation with given controller and trajectory
        self.Tmax = self.traj.TS[-1]

        print("sleeping for 10 sec ....")
        time.sleep(10)
        self.t0 = self.get_clock().now().nanoseconds
        print("t0:", self.t0, "...... Tmax: ", self.Tmax, "\n")

        #trajectory publisher
        self.traj_publisher = self.create_publisher(TrajectorySetpoint, 'mujoco_px4/trajectory_setpoint', 10)

        #timer 
        self.timer = self.create_timer(0.005, self.trajectory_publisher)


    def trajectory_publisher(self): #runs at 0.02 Hz
        t = (self.get_clock().now().nanoseconds - self.t0)/10**9
        # print("t: ", t, "\n")

        #timestamp conditional to publish
        if(t< self.Tmax):
            des_state = self.traj.get_des_state(t)

            #define the msg
            msg = TrajectorySetpoint()
            # msg.timestamp = t
            msg.position = [des_state.pos[0], des_state.pos[1], des_state.pos[2]]
            msg.velocity = [des_state.vel[0], des_state.vel[1], des_state.vel[2]]
            msg.yaw = 0.0
            # msg.yaw = des_state.yaw

            #publish msg
            self.traj_publisher.publish(msg)
        else:
            self.get_logger().info("finished trajectory publishing. Reached goal!")
            # self.traj_publisher.publish(msg)







def main(args = None):
    rclpy.init(args=args)
    trajectory_planner = TrajectoryPlannerRRT()
    rclpy.spin(trajectory_planner)
    trajectory_planner.destroy_node()
    rclpy.shutdown()



if __name__=="__main__":
    main()