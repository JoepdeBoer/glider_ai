import numpy as np
import matplotlib.pyplot as plt

g = 9.81

class Agent():
    def __init__(self, x, y, z, V, theta,
                 s1, v1, s2, v2, s3,
                 v3, vstall, vne, objective, time_step = 1, name= 'Agent'):
        """
        x, y, z in m
        All speeds in m/s
        bank in rad
        heading in rad
        """
        self.x = x
        self.y = y
        self.z = z
        self.V = V
        self.b = 0
        self.db = 0
        self.dV = 0
        self.theta = theta
        self.vstall = vstall
        self.vne = vne
        self.polar = np.polyfit([v1, v2, v3], [s1, s2, s3], 2)
        a = self.polar[0]
        b = self.polar[1]
        c = self.polar[2]
        V = np.sqrt(c/a)
        d = 2 * a * V + b
        self.bestLD = -1/d
        self.name = name
        self.time_step = time_step
        self.objective = objective # shape (2,)
        self.history = []

    def get_state(self, wind):
        dict = {'xyz': np.array([self.x, self.y, self.z], dtype=np.float32),
                'V': self.V,
                'thetha': self.theta, # in radian[self.theta],
                'b': self.b, # in rad
                'updraft': wind.updraft(self.x, self.y, self.z),
                'objective': self.objective}
        return dict

    def move(self, Windfield):
        self.history.append([self.x, self.y, self.z, self.V, self.b, self.theta])

        updraft = Windfield.updraft(self.x, self.y, self.z)
        self.x = self.x + self.V*np.cos(self.theta) * self.time_step
        self.y = self.y - self.V*np.sin(self.theta) * self.time_step
        self.z = self.z + self.sink()*self.time_step + updraft*self.time_step - self.dV * (self.dV + 2 * self.V)/(2 * 9.81)

        if self.V + self.dV > 1.2 * self.vstall and self.V + self.dV < self.vne:
            self.V = self.V + self.dV # only changing V when it is far enough from stall
        if self.b + self.db < np.pi/3 and self.b + self.db > -np.pi/3:
            self.b = self.b + self.db
        self.theta = self.theta + self.dtheta()


    def take_action(self, db, dV):
        """Taking action given as single value arrays to floats
        db is scaled to the rolling rate of a discus flying at 90-100km/h
        dv is scaed to have a maximum of +- 3 m/s^2 """

        self.db = float(db) * 90/180 * np.pi / 4.2 * self.time_step
        self.dV = float(dV) * 3 * self.time_step



    def sink(self):
        """
        Calculates the sink of the plane
        Taking into account the polar
        and a shift due to the bank angle increasing load factor
        assumes best LD is constant and polar shifts along best LD direction

        """
        n = 1/np.cos(self.b)    # load factor
        V2 = np.sqrt(self.polar[2] * n / self.polar[0]) # V for best L/D at this loading

        # shift to get new polar
        hshift = V2 - np.sqrt(self.polar[2] / self.polar[0]) # horizontal shift
        vshift = hshift/(-self.bestLD) # vertical shift

        V_eval = self.V - hshift # Velocity to evaluate new sink
        sink = np.polyval(self.polar, V_eval) + vshift # actual sink of plane
        return sink

    def dtheta(self):
        return np.tan(self.b) * g / self.V * self.time_step

    def plot_history(self):
        x = [i[0] for i in self.history]
        y = [i[1] for i in self.history]
        z = [i[2] for i in self.history]
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x, y, z)
        plt.show()
