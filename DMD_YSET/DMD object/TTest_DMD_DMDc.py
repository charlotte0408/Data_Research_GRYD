from DMD_func import *
from numpy import set_printoptions
from scipy.stats import ttest_ind

# Data Path
path = raw_input('Enter filename of YRR Dataframe: ')

n = int(raw_input('Enter number of sampling replication: '))
t = float(raw_input('Enter proportion of training set (0-1): '))

# ---------- Original DMD ----------
wait = raw_input('Press Enter to Start DMD Sampling: ')
        
# Data Preprocessing
Y1,R1,R2,sub = DMD_YRR_prep(path)

set_printoptions(threshold=maxsize)
DMDO = np.empty((0,2))

# Sample and run for 100 times
for m in range(n):

    Y1_R11 = DMD(Y1, R1, sub, train = 0.8, name = "Y1-R1 Original DMD")
    Y1_R12 = DMD(Y1, R1, sub, train = 0.8, pred = 'Whole', name = "R1-R2 Original DMD")
    sum1 = Y1_R11.MSE
    sum2 = Y1_R12.MSE

    DMDO = np.vstack([DMDO,[sum1,sum2]])
    
    if m % 10 == 9:
        print('DMD Trial %d Finished ' % (m+1))
    
# ---------- DMD with Control ----------
control = ['Gender', 'Ethnicity', 'GRYD_Zone', 'Age']

for ctrl in control:

    DMDc = """wait = raw_input('Press Enter to Start DMDc-{0} sampling: ')

# Data Preprocessing
Y1,R1,R2,sub = DMD_YRR_prep(path, control = '{0}')

DMDc_{0} = np.empty((0,2))

for m in range(n):

    Y1_R11 = DMD(Y1, R1, sub, train = 0.8, control = '{0}', name = "Y1-R1 DMDc on {0}")
    Y1_R12 = DMD(Y1, R1, sub, train = 0.8, control = '{0}', pred = 'Whole', name = "Y1-R1 DMDc on {0}")

    sum1 = Y1_R11.MSE
    sum2 = Y1_R12.MSE

    if sum1<=1 and sum2<=1:
        DMDc_{0} = np.vstack([DMDc_{0},[sum1,sum2]])
        
        if m % 10 == 9:
            print('DMDc {0} Trial %d Finished ' % (m+1))
    else:
        m -= 1""".format(ctrl)
    
    exec(DMDc)


# ---------- T-test ---------- 
wait = raw_input('Press Enter to Start T-Test: ')

for ctrl in control:
    
    t_test = """Test_T,Test_P = ttest_ind(DMDO[:,0],DMDc_{0}[:,0])
Whole_T,Whole_P = ttest_ind(DMDO[:,1],DMDc_{0}[:,1])
print('Stats of The Testing Set ({0}):')
print('    t-stats: %.4f ' % Test_T)
print('    p-value: %.4f ' % Test_P)
print('Stats of The Whole Set ({0}):')
print('    t-stats: %.4f ' % Whole_T)
print('    p-value: %.4f ' % Whole_P)""".format(ctrl)
    exec(t_test)

