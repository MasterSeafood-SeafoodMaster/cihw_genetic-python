import Car_toolkit as ct
import numpy as np

class RBFN:
    def __init__(self, input_dim, hidden_dim, output_dim, sigma=40):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma
        
        self.centers = None
        self.weights = None
        
    def _rbf(self, X, centers):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)
        return np.exp(-distances**2 / (2*self.sigma**2))
    
    def train(self, X, y):
        # Initialize centers using K-means clustering
        idx = np.random.choice(len(X), self.hidden_dim, replace=False)
        self.centers = X[idx]
        #print(self.centers)
        
        # Compute RBF outputs for each data point and center
        RBF_outputs = self._rbf(X, self.centers)
        
        # Solve for weights using linear regression
        self.weights = np.linalg.lstsq(RBF_outputs, y, rcond=None)[0]

    def getDNA(self):
        #print(self.centers)
        self.DNA = np.concatenate((self.weights, np.reshape(self.centers, (30,)) ))
        return self.DNA

    def setDNA(self, DNA):
        self.weights, self.centers = np.split(DNA, [10])
        self.centers = np.reshape(self.centers, (10, 3))

    def predict(self, X):
        RBF_outputs = self._rbf(X, self.centers)
        y_pred = np.dot(RBF_outputs, self.weights)
        return y_pred

def computeScore(DNA):
    rbfn = RBFN(input_dim=3, hidden_dim=10, output_dim=1)
    rbfn.setDNA(DNA)

    square, car_pos, endline = ct.getSquare("軌道座標點.txt")
    frame=0
    s=0
    while (not(ct.inBox(car_pos[0], car_pos[1])))and frame<100:
        sX = square[:, 0]; sY = square[:, 1]
        try:
            sensors, min_ds, min_ds_point = ct.draw_sensors(car_pos[0], car_pos[1], car_pos[2], square)
        except:
            break
        rd, fd, ld = min_ds
        sensors_np = np.array([[fd, rd, ld]])
        theta = rbfn.predict(sensors_np)[0]
        car_pos = ct.nextPos(car_pos[0], car_pos[1], car_pos[2], theta)
        #d = rd+fd+ld
        #s+=d

        if ct.inBox(car_pos[0], car_pos[1]):
            return(300-frame)
        frame+=1
    return frame
    

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :3].astype(float) # input sensor readings
    y = data[:, 3].astype(float) # corresponding steering wheel angle
    return X, y

"""
X, y = load_data("./train4dAll.txt")

rbf = RBFN(input_dim=3, hidden_dim=10, output_dim=1)
rbf.train(X, y)

X_test = np.array([[24.0163, 07.9606, 09.0731]])
y_pred = rbf.predict(X_test)
print(y_pred)
"""

