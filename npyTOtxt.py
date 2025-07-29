import sys
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import copy
from numpy.random import default_rng
from timeit import default_timer as timer
from pdb import set_trace
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# design of standard DNN; architecture can be changed easily for testing structural influence
class DNN(torch.nn.Module):
    def __init__(self, architecture):
        super(DNN, self).__init__()
        # deploy layers
        self.layers = torch.nn.Sequential(architecture)
    def forward(self, x):
        out = self.layers(x)
        return out

# design for multi material version of PINN
class PINN_Source_MATS_LBFGS():
    def __init__(self, architecture, constants, tckls, dose, x_i, x_bc, x_d, t_i, t_bc, t_d, temp_i, temp_bc, temp_d, x_f, t_f, plot_grad=False, a1=1, a2=1, a3=1, a4=0, weightScale=0):
        
        # assign tensors already containing grad=True when applicable
        self.x_i = x_i
        self.x_bc = x_bc
        self.x_d = x_d
        self.x_f = x_f
        self.temp_i = temp_i
        self.temp_bc = temp_bc
        self.temp_d = temp_d
        self.t_i = t_i
        self.t_bc = t_bc
        self.t_d = t_d
        self.t_f = t_f
        self.dose = dose

        # apply architecture to the standard DNN
        self.dnn = DNN(architecture)
        self.dnn.to(device)

        # store lists and variables for tracking loss as training occurs
        self.iter = 0
        self.loss_i_list = list()
        self.loss_bc_list = list()
        self.loss_d_list = list()
        self.loss_f_list = list()
        self.loss_list = list()
        
        self.plot_grad = plot_grad
        self.ave_grads = list()
        self.layer_names = list()
        
        # define constants function
        def get_const(x, const, tckls):
            alpha_list = list()
            nu_list = list()
            thresh = list()
            if type(tckls) == float or type(tckls) == int:
                thresh = None
            else:
                for i in range(len(tckls)):
                    if i == 0:
                        thresh.append(tckls[i])
                    else:
                        thresh.append(np.sum(tckls[:i+1]))
                thresh.pop(-1)
                const = const[:len(tckls)]
            print(f'Threshold values: {thresh}', flush=True)
            if type(thresh) == type(None):
                alpha_list = [const[0][0]] * int(len(x))
                nu_list = [const[0][1]] * int(len(x))
            else:
                for i in range(len(x)):
                    if len(thresh) == 0:
                        alpha_list.append(const[0][0])
                        nu_list.append(const[0][1])
                        continue
                    if x[i].item() < thresh[0]:
                        alpha_list.append(const[0][0])
                        nu_list.append(const[0][1])
                        continue
                    if x[i].item() >= thresh[-1]:
                        alpha_list.append(const[-1][0])
                        nu_list.append(const[-1][1])
                        continue
                    for j in range(len(thresh)):
                        if x[i].item() >= thresh[j] and x[i].item() < thresh[j+1]:
                            alpha_list.append(const[j+1][0])
                            nu_list.append(const[j+1][1])
                            break

            alpha = torch.tensor(np.array(alpha_list).reshape(-1,1)).float()
            nu = torch.tensor(np.array(nu_list).reshape(-1,1)).float()
            alpha = alpha.to(device)
            nu = nu.to(device)

            return alpha, nu, thresh

        self.alpha, self.nu, self.thresh = get_const(self.x_f, constants, tckls)
        
        if weightScale < 1 and weightScale > 0 and a3 == 0:
            self.a1 = a1
            self.a2 = a2
            self.a3 = a3
            self.a4 = a4    
        else:
            MB = torch.mean(self.temp_bc ** 2) * a1
            MI = torch.mean((self.nu * self.dose) ** 2) * a2
            MT = torch.mean(self.temp_i ** 2) * a3
            MD = torch.mean(self.temp_d ** 2) * a4
            N = a1 + a2 + a3 + a4
            Msum = MI + MB + MT + MD
            self.a1 = a1 * (1 - MB / Msum) / (N - 1)
            self.a2 = a2 * (1 - MI / Msum) / (N - 1)
            self.a3 = a3 * (1 - MT / Msum) / (N - 1)
            self.a4 = a4 * (1 - MD / Msum) / (N - 1)     
           

        # initiate L-BFGS optimizezr
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe")

    # Neural Network prediction function for initial and boundary conditions
    def NN(self, x, t):  
        return self.dnn(torch.cat([x, t], dim=1))
    
    # differential equation to be optimized
    def f(self, x, t):
        # calculate the derivatives with automatic differentiation
        u = self.NN(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True,create_graph=True)[0]
        u_xx = u_xx.to(device)

        # return the differential equation to be optimized
        return u_t - (self.alpha * u_xx) 
        
    def loss_func(self):
        self.optimizer.zero_grad()

        d_pred = self.NN(self.x_d, self.t_d)
        i_pred = self.NN(self.x_i, self.t_i)
        bc_pred = self.NN(self.x_bc, self.t_bc)
        f_pred = self.f(self.x_f, self.t_f)

        loss_i = torch.mean((self.temp_i - i_pred) ** 2)
        loss_bc = torch.mean((self.temp_bc - bc_pred) ** 2)
        loss_d = torch.mean((self.temp_d - d_pred) ** 2)
        loss_f = torch.mean((f_pred - self.nu * self.dose) ** 2)        
       
        w1 = self.a1
        w2 = self.a2
        w3 = self.a3
        w4 = self.a4

        loss = w1 * loss_bc + w2 * loss_f + w3 * loss_i + w4 * loss_d

        loss.backward()
        
        if self.plot_grad:
            self.plot_grad_flow(self.dnn.named_parameters())
        # append loss values to their respective lists
        self.loss_list.append(loss.item())
        self.loss_i_list.append(loss_i.item())
        self.loss_bc_list.append(loss_bc.item())
        self.loss_d_list.append(loss_d.item())
        self.loss_f_list.append(loss_f.item())

        self.iter += 1
        if self.iter % 100 == 0:
            print('Iter {}, Loss: {}, Loss_bc: {}, Loss_f: {}, Loss_i: {}, Loss_d: {}, w1: {}, w2: {}, w3: {}, w4: {}'.format(self.iter, loss.item(), loss_bc.item(), loss_f.item(), loss_i.item(), loss_d.item(), w1, w2, w3, w4), flush=True)
        
        if self.iter < 100:
            print('Iter {}, Loss: {}, Loss_bc: {}, Loss_f: {}, Loss_i: {}, Loss_d: {}, w1: {}, w2: {}, w3: {}, w4: {}'.format(self.iter, loss.item(), loss_bc.item(), loss_f.item(), loss_i.item(), loss_d.item(), w1, w2, w3, w4), flush=True)

        return loss
    
    def train(self):
        self.optimizer.step(self.loss_func)


#Program Input
if len(sys.argv) > 1:
    seed = sys.argv[1]
    name = sys.argv[2]
    a1Input = float(sys.argv[3])
    a2Input = float(sys.argv[4])
    a3Input = float(sys.argv[5])
    weightScale= float(sys.argv[6])
    layers = int(sys.argv[7])
    nodes  = int(sys.argv[8])
    model = sys.argv[9]
else:
    seed = 1
    name = 1
    a1Input = 1
    a2Input = 1
    a3Input = 0
    weightScale = 0
    layers = 4
    nodes = 64
    model = '1bXL'
    filepath = 'C:/jennydocs/code/pinn/'


print(a1Input)
print(a2Input)
print(a3Input)
print(weightScale)
print(layers)
print(nodes)
print(model)

if weightScale > 1:     
    N_d = int(sys.argv[6])
    tstr = N_d
    dataWeight = 1
else:
    N_d = 1
    tstr = weightScale
    dataWeight = 0

if weightScale < 1 and weightScale > 0 and a3Input == 0:
    a1Input = 1 - weightScale
    a2Input = weightScale

filename = str(name) + "_a1_" + str(a1Input) + "_a2_" + str(a2Input) + "_a3_" + str(a3Input) + "_t_" + str(tstr) + "_L_" + str(layers) + "_N_" + str(nodes) + "_S_" + str(model)

rng = default_rng(int(seed))
torch.cuda.manual_seed_all(int(seed))
torch.manual_seed(int(seed))

if torch.cuda.is_available():
    print("true", flush=True);
else:
    print("false", flush=True);

# Data Parameters for Training and Testing
N_f = 25_000 # num collocation points
N_i = 250
N_bc = 500

N_test = 100_000

# Model Parameters
structure = list()
structure.append(2)
for cNode in range(layers):
    structure.append(nodes)
structure.append(1)

activation_fn = torch.nn.Tanh 
scale = 10

# Data Load

Temp = np.load(filepath + 'temp_array_' + str(model) + '.npy', allow_pickle=True)
dose = np.load(filepath + 'dose_array_' + str(model) + '.npy', allow_pickle=True)
location = np.load(filepath + 'locations_array_' + str(model) + '.npy', allow_pickle=True)

epidermis = 	[5.485527544351074e-09, 2.334267040149393e-07]
dermis =	[1.2137577394103048e-07, 2.7275454818209095e-07]
fat = 		[7.056490063699127e-08, 3.814318953350879e-07]

tckls = [0.0016, 0.0104]

if str(model) == '2f':
    depths = [0.0016, 0.0104]
    constants = [epidermis, fat]

if str(model) == '2b':
    depths = [0.0016, 0.0104]
    constants = [dermis, fat]

if str(model) == '2bXL':
    depths = [0.0016, 0.0104]
    constants = [dermis, fat]

if str(model) == '2ba':
    depths = [0.001, 0.011]
    constants = [dermis, fat]

if str(model) == '2bc':
    depths = [0.0036, 0.0084]
    constants = [dermis, fat]

if str(model) == '2bb':
    depths = [0.0026, 0.0094]
    constants = [dermis, fat]

if str(model) == '3b':
    depths = [0.0015, 0.0025, 0.008]

if str(model) == '2baXL':
    depths = [0.001, 0.011]
    constants = [dermis, fat]

if str(model) == '2bcXL':
    depths = [0.0036, 0.0084]
    constants = [dermis, fat]

if str(model) == '2bbXL':
    depths = [0.0026, 0.0094]
    constants = [dermis, fat]

if str(model) == '3bXL':
    depths = [0.0015, 0.0025, 0.008]

if str(model) == '1':
    depths = [0.012]
    constants = [dermis]

if str(model) == '1q':
    depths = [0.012]
    constants = [dermis]

if str(model) == '1a':
    depths = [0.012]
    constants = [epidermis]

if str(model) == '1b':
    depths = [0.012]
    constants = [fat]


if str(model) == '1t':
    depths = [0.012]
    constants = [dermis]

if str(model) == '1at':
    depths = [0.012]
    constants = [epidermis]

if str(model) == '1bt':
    depths = [0.012]
    constants = [fat]




tlen = location.shape[0]
xlen = location.shape[1]
t = np.arange(0, tlen, 1) / tlen
x = location[0] / 1000
X,T = np.meshgrid(x,t)

flatTemp = Temp.flatten()
flatT = T.flatten()
flatX = X.flatten()
flatDose = dose.flatten()
testIndex = rng.choice(X.size, N_test, replace=False)

mask = np.full(X.size, True, dtype=bool)
mask[testIndex] = False

if seed == 1:


    np.savetxt('temp_' + str(model) + '.txt', flatTemp)
    np.savetxt('t_' + str(model) + '.txt', flatT)
    np.savetxt('x_' + str(model) + '.txt', flatX)
    np.savetxt('dose_' + str(model) + '.txt', flatDose)


    np.savetxt('tempTest_' + str(model) + '.txt', flatTemp[testIndex])
    np.savetxt('tTest_' + str(model) + '.txt', flatT[testIndex])
    np.savetxt('xTest_' + str(model) + '.txt', flatX[testIndex])
    np.savetxt('doseTest_' + str(model) + '.txt', flatDose[testIndex])
    np.savetxt('tempTrain_' + str(model) + '.txt', flatTemp[mask])
    np.savetxt('tTrain_' + str(model) + '.txt', flatT[mask])
    np.savetxt('xTrain_' + str(model) + '.txt', flatX[mask])
    np.savetxt('doseTrain_' + str(model) + '.txt', flatDose[mask])


# Initial Conditions
ixt = np.hstack((x.reshape(xlen, 1), np.zeros([xlen, 1])))
itemp = Temp[0].reshape(-1,1)
idose = dose[0].reshape(-1,1)

idx = rng.choice(ixt.shape[0], N_i, replace=False)
ixt = ixt[idx, :]
itemp = itemp[idx, :]
idose = idose[idx, :]

# Boundaries
i = np.ndarray.flatten(np.zeros([tlen, 1])).reshape(tlen, 1)
bound1 = np.hstack((i, t.reshape(tlen, 1)))
j = np.ndarray.flatten(np.ones([tlen, 1])).reshape(tlen, 1)
bound2 = np.hstack((j * x[xlen - 1], t.reshape(tlen, 1)))
bcxt = np.vstack((bound1, bound2))

bctemp = np.vstack((Temp[:,0:1], Temp[:,-1:]))
bcdose = np.array([dose[0][0]] * int(len(bcxt)/4)).reshape(-1,1)
bcdose = np.vstack((bcdose, np.array([dose[0][-1]]  * int(len(bcxt)/4)).reshape(-1,1)))
bcdose = np.vstack((bcdose, np.array([dose[-1][0]]  * int(len(bcxt)/4)).reshape(-1,1)))
bcdose = np.vstack((bcdose, np.array([dose[-1][-1]] * int(len(bcxt)/4)).reshape(-1,1)))
idx = rng.choice(bcxt.shape[0], N_bc, replace=False)
bcxt = bcxt[idx, :]
bctemp = bctemp[idx, :]
bcdose = bcdose[idx, :]

if a3Input == 0:
   bcxt = np.vstack([ixt, bcxt])
   bctemp = np.vstack([itemp, bctemp])
   bcdose = np.vstack([idose, bcdose])


# Function
# allow replacement because dose is involved
idxx = rng.choice(x.shape[0], N_f, replace=True)
idxt = rng.choice(t.shape[0], N_f, replace=True)
x_f_tr = x[idxx.astype(int)].reshape(-1,1)
t_f_tr = t[idxt.astype(int)].reshape(-1,1)
col_points = np.hstack((x_f_tr, t_f_tr))
dose_f = dose[idxt, idxx]

# Data
idxx = rng.choice(x.shape[0], N_d, replace=True)
idxt = rng.choice(t.shape[0], N_d, replace=True)
x_d = x[idxx.astype(int)].reshape(-1,1)
t_d = t[idxt.astype(int)].reshape(-1,1)
temp_d = Temp[idxt, idxx]
temp_d = temp_d.reshape(len(temp_d), 1)
x_d = torch.tensor(x_d, requires_grad=True).float()
t_d = torch.tensor(t_d, requires_grad=True).float()
temp_d = torch.tensor(temp_d).float()


# combine collocation points and initial and boundary condition points to include icbc points in col points

if a3Input == 0:
    col_points = np.vstack((col_points, bcxt))
    col_dose = np.vstack((dose_f.reshape(-1, 1), bcdose))
else:
    col_points = np.vstack((col_points, bcxt))
    col_dose = np.vstack((dose_f.reshape(-1, 1), bcdose))
    col_points = np.vstack((col_points, ixt))
    col_dose = np.vstack((col_dose, idose))

# separate x from t and create tensors from them, with grad when applicable
x_i = torch.tensor(ixt[:, 0:1], requires_grad=True).float()
t_i = torch.tensor(ixt[:, 1:2], requires_grad=True).float()
temp_i = torch.tensor(itemp).float()
x_bc = torch.tensor(bcxt[:, 0:1], requires_grad=True).float()
t_bc = torch.tensor(bcxt[:, 1:2], requires_grad=True).float()
temp_bc = torch.tensor(bctemp).float()
x_f = torch.tensor(col_points[:, 0:1], requires_grad=True).float()
t_f = torch.tensor(col_points[:, 1:2], requires_grad=True).float()
dose = torch.tensor(col_dose).float()

x_i = x_i.to(device)
t_i = t_i.to(device)
x_bc = x_bc.to(device)
t_bc = t_bc.to(device)
x_f = x_f.to(device)
t_f = t_f.to(device)
x_d = x_d.to(device)
t_d = t_d.to(device)
temp_i = temp_i.to(device)
temp_bc = temp_bc.to(device)
temp_d = temp_d.to(device)
dose = dose.to(device)

# ---------------------------------------------------------------------------------------------------------- #
# Model and Method Preparation and Training
# ---------------------------------------------------------------------------------------------------------- #
init_temp = Temp[0, ]
temp_i = temp_i - init_temp[0]
temp_i = temp_i/scale
temp_bc = temp_bc - init_temp[0]
temp_bc = temp_bc/scale
temp_d = temp_d - init_temp[0]
temp_d = temp_d / scale
dose = dose/scale


# loop through structure and form dictionary of desired architecture
layers = list()
for i in range(len(structure)-2):
    layers.append((f'layer_{i+1}', torch.nn.Linear(structure[i], structure[i+1])))
    layers.append((f'activation_{i+1}', activation_fn()))
layers.append((f'layer_{len(structure)-1}', torch.nn.Linear(structure[-2], structure[-1])))
architecture = OrderedDict(layers)

# initiate model, optimizer, and iteration count

model = PINN_Source_MATS_LBFGS(architecture, constants, tckls, dose, x_i, x_bc, x_d, t_i, t_bc, t_d, temp_i, temp_bc, temp_d, x_f, t_f, plot_grad=False, a1 = a1Input, a2 = a2Input, a3 = a3Input, a4 = dataWeight, weightScale = weightScale)
model.dnn.train()
model.train()
model.dnn.eval()


# test model
X_test_orig = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_test_orig = Temp.flatten()[:,None]
idx_test = rng.choice(X_test_orig.shape[0], N_test, replace=False)
X_test = X_test_orig[idx_test, :]
u_test = u_test_orig[idx_test, :]

X_test = torch.tensor(X_test).float()
X_test = X_test.to(device)
prediction = model.dnn(X_test).detach().cpu().numpy()

prediction = prediction * scale
prediction = prediction + init_temp[0]
prediction_df = pd.DataFrame(prediction, columns=['Guess'])
prediction_df['True'] = u_test

results = filename  + "_results.dat"
with open(results, 'a') as f:
    df_string = prediction_df.to_string()
    f.write(df_string)

end_score = r2_score(u_test, prediction)



inputs = torch.tensor(X_test_orig).float()
inputs = inputs.to(device)
predict = model.dnn(inputs).detach().cpu().numpy()
predict = predict * scale
predict = predict + init_temp[0]
results = filename  + "_plotresults.dat"
np.savetxt(results, predict)


print()  
print()
print(f'Score: {end_score}')
