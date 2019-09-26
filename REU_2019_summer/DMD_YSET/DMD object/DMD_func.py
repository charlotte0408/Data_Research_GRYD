import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sys import maxsize
import copy

def DMD_YRR_prep(YRR_path, control = 'None'):
    

    '''

    DMD_YRR_prep

        @brief A function that prepares matrices for DMD calculations

        @param <str> YRR_path: Path to a YRR dataframe

                                    Need to have following attributes:

                                    - YSET_I_R
                                    - YSETResults
                                    - a1_combo,...,t39_combo

                                    Note: Each UniqueID should have exactly 3 responses - 1 intake and 2 corresponding first retakes.

        @return <List>  <np.ndarray> Y1  - Intake Matrix
                        <np.ndarray> R1  - Retake 1 Matrix
                        <np.ndarray> R2  - Retake 2 Matrix
                        <int>        sub - number of control rows added


    '''

    ldict = {}
    
    # -------- Read Data --------   
    df_c= pd.read_csv(YRR_path)
    df_c.drop('Unnamed: 0', axis = 1, inplace = True)
    Y1_idx=range(0,df_c.shape[0],3)
    R1_idx=range(1,df_c.shape[0],3)
    R2_idx=range(2,df_c.shape[0],3)

    var_n = ['Y1', 'R1', 'R2']

    newlist = []
    for i in range(9):
        newlist.append(i)
    for i in range(16, 39):
        newlist.append(i)
    
    # -------- Prepare Dataframe for each timestamp --------
    # *** Might need to adjust if additional controls are introduced!! ***
    for var in var_n:
        read_data = """import numpy as np
import pandas as pd
import copy
df_{0}=df_c.iloc[{0}_idx,:]
dfr = copy.deepcopy(df_{0}.loc[:, 'RiskFactor'])
    
if control != 'None':
    if control == 'Ethnicity':
        dfi = copy.deepcopy(df_{0}.loc[:,'Ethnicity_Asian':'Ethnicity_Other'])
    else:
        dfi = copy.deepcopy(df_{0}.loc[:, '{1}'])

df_{0} = df_{0}.loc[:,'a1_combo':'t39_combo']
df_{0} = pd.concat([df_{0}, dfr], axis=1)
    
if control != 'None':
    df_{0} = pd.concat([df_{0}, dfi], axis=1)
    
{0} = df_{0}.convert_objects(convert_numeric=True).values
{0} = {0}.transpose()
""".format(var, control)
        exec(read_data,locals(),ldict)
        
    Y1 = ldict['Y1']
    R1 = ldict['R1']
    R2 = ldict['R2']
    
    # -------- One-hot coding for each control accordingly --------
    # *** Might need to adjust if additional controls are introduced!! ***
    if control != 'None': 

        m = int(Y1.shape[0]-1)
        n = int(Y1.shape[1])
        for i in range(Y1.shape[1]):
            if R1[m,i]!=Y1[m,i]:
                R1[m,i] = Y1[m,i]
            if R2[m,i]!=Y1[m,i]:
                R2[m,i] = Y1[m,i]


        List_Num = []
        for i in range(Y1.shape[1]):
            List_Num.append(Y1[40,i])
            List_Num.append(R1[40,i])
            List_Num.append(R2[40,i])
        List_Num = list(dict.fromkeys(List_Num))
        Numpy_Num = np.array(List_Num)
        Numpy_Num = np.sort(Numpy_Num)

        for var in var_n:
            onehot = """
Add_Matrix_{0} = np.zeros((len(List_Num),n))

if control == 'Ethnicity':
    sub = 5
    
else:
    if control == 'GRYD_Zone':
        for i in range({0}.shape[1]):
            if {0}[{0}.shape[0]-1,i]<=6.0:
                k = {0}[-1,i]-1
                Add_Matrix_{0}[int(k),i] = 1.0
            elif {0}[{0}.shape[0]-1,i]>6.0 and {0}[{0}.shape[0]-1,i]<=16.0:
                k = {0}[-1,i]-2
                Add_Matrix_{0}[int(k),i] = 1.0
            else:
                k = {0}[-1,i]-3
                Add_Matrix_{0}[int(k),i] = 1.0
        {0} = np.delete({0},-1,axis=0)
        {0} = np.vstack([{0},Add_Matrix_{0}])
        sub = 21

    else: 
        min_val = np.amin({0}[m,:])
        max_val = np.amax({0}[m,:])
        Add_Matrix = np.zeros((int(max_val-min_val+1),n))
        for i in range({0}.shape[1]):
            k = int({0}[-1,i]-min_val)
            Add_Matrix[k,i] = 1.0
        {0} = np.delete({0},-1,axis=0)
        {0} = np.vstack([{0},Add_Matrix])
        sub = int(max_val-min_val+1)""".format(var)
            exec(onehot,locals(),ldict)
        sub = ldict['sub']
    else:
        sub = 0
    
    # -------- Standardization --------
    
    for var in var_n:
        std = """
for i in range({0}.shape[1]):
    {0}[37][i] += 1 
    {0}[38][i] += 1


for i in newlist:
    for j in range({0}.shape[1]):
        {0}[i][j] = ({0}[i][j] - 1) / 4

for i in range({0}.shape[1]):
    {0}[39][i] /= 9""".format(var)
        exec(std,locals(),ldict)
        
    Y1 = ldict['Y1']
    R1 = ldict['R1']
    R2 = ldict['R2']
        
    return [Y1, R1, R2, int(sub)]





class DMD:
    
    def __init__(self, X0, X1, sub = 0, train = 1.0, test = 0, control = 'None', pred = 'Test', name = "DMD"):    
    
        '''

        DMD

            @brief Dynamic Mode Decomposition that supports DMDc

            @param  <np.ndarray> X0               - Matrix at timestamp 0
                    <np.ndarray> X1               - Matrix at timestamp 1
                    <int>        sub = 0          - Number of control variables added, default is 0
                    <float>      train = 1.0      - Proportion of training set, value should be between 0 and 1, default is 1
                    <float>      test = 0         - Proportion of the testing set, value should between 0 and 1, default is 0
                    <str>        control = 'None' - Name of control variable, default is 'None'
                    <str>        pred = 'Test'    - Dataset to predict on, value should be either 'Test' or 'Whole', default is 'Test'

            @return <List>  <np.ndarray> A_X0_X1  - Transition Matrix approximated by DMD 
                            <float>      MSE      - Mean square error of the test result


        '''
        
        self.name = name

        if train + test != 1:
            if train == 1 and test != 0:
                train = 1 - test


        # Test-train Split
        Index = list(range(X0.shape[1]))
        random.shuffle(Index)
        Train_X = np.empty((X0.shape[0], 0))
        Test_X = np.empty((X0.shape[0], 0))
        Train_Y = np.empty((X1.shape[0] - sub, 0))
        Test_Y = np.empty((X1.shape[0] - sub, 0))
        for i in range(int(round(len(Index) * train))):
            Train_X = np.hstack((Train_X, X0[:, Index[i]].reshape(-1, 1)))
            if sub != 0:
                Train_Y = np.hstack((Train_Y, X1[:-sub, Index[i]].reshape(-1, 1)))
            else:
                Train_Y = np.hstack((Train_Y, X1[:, Index[i]].reshape(-1, 1)))
        for i in range(int(round(len(Index) * train)), len(Index)):
            Test_X = np.hstack((Test_X, X0[:, Index[i]].reshape(-1, 1)))
            if sub != 0:
                Test_Y = np.hstack((Test_Y, X1[:-sub, Index[i]].reshape(-1, 1)))
            else:
                Test_Y = np.hstack((Test_Y, X1[:, Index[i]].reshape(-1, 1)))

        # SVD for X0
        U_X0, Sig_X0, V_X0 = np.linalg.svd(Train_X, full_matrices=False)
        U_X0_T = U_X0.conjugate().transpose()
        V_X0_T = V_X0.conjugate().transpose()
        Sig_inv_X0 = np.zeros((X0.shape[0], X0.shape[0]))
        for i in range(X0.shape[0]):
            for j in range(X0.shape[0]):
                if i == j:
                    Sig_inv_X0[i][j] = 1 / Sig_X0[i]

        # Build up the DMD A matrix for X0 and X1
        A_step1 = np.dot(Train_Y, V_X0_T)
        A_step2 = np.dot(A_step1, Sig_inv_X0)
        A_X0_X1 = np.dot(A_step2, U_X0_T)     

        if pred == 'Whole':
            Pred = np.dot(A_X0_X1, X0)
            if sub != 0:
                MSE = mean_squared_error(X1[:-sub, :], Pred)
            else: MSE = mean_squared_error(X1, Pred)
        else:
            Pred = np.dot(A_X0_X1, Test_X)
            MSE = mean_squared_error(Test_Y, Pred)

        self.A = A_X0_X1
        self.control = control
        self.MSE = MSE
        self.RF = A_X0_X1[39,:39]
        if control != 'None': 
            self.eigval, self.eigvec = np.linalg.eig(A_X0_X1[:, :-sub])
        else:
            self.eigval, self.eigvec = np.linalg.eig(A_X0_X1)
        self.w_domeigvec = self.eigvec[:,0].real[:39]
        self.eiglog = np.log(self.eigval)[:39]
        self.label = {'a1':'a1: I try to be nice to other people because I care about their feelings.', 
                      'a2':'a2: I get very angry and "lose my temper" (yell or get mad).',
                      'a3':'a3: I do as I am told.',
                      'a4':'a4: I try to scare people to get what I want.',
                      'a5':'a5: I am accused of not telling the truth or cheating.', 
                      'a6':'a6: I take things that are not mine from home, school, or elsewhere.',
                      'b7':'b7: When I go out, I tell my parents or guardians where I am going or leave them a note.',
                      'b8':'b8: My parents or guardians know where I am when I am not at home or at school.',
                      'b9':'b9: My parents or guardians know who I am with, when I am not at home or at school.',
                      'c10':'c10: Did you fail to go on to the next grade in school or fail a course in school?',
                      'c11':'c11: Did you get suspended, expelled or transferred to another school for disciplinary reasons?',
                      'c12':'c12: Did you "go out" on a date with a boyfriend or girlfriend for the very first time?', 
                      'c13':'c13: Did you break up with a boyfriend or girlfriend or did he or she break up with you?',
                      'c14':'c14: Did you have a big fight or problem with a friend?', 
                      'c15':'c15: Did you start hanging out with a new group of friends?',
                      'c16':'c16: Did anyone you were close too die or get seriously injured?',
                      'de17':'de17: Sometimes I like to do something a little dangerous just for the fun of it.',
                      'de18':'de18: I sometimes find it exciting to do things that might get me in trouble.',
                      'de19':'de19: I often do things without stopping to think if I will get in trouble for it.',
                      'de20':'de20: I like to have fun when I can, even if I will get into trouble for it later.',
                      'f21':'f21: It is okay for me to lie (or not tell the truth) if it will keep my friends from getting in trouble with \n       parents, teachers or police.', 
                      'f22':'f22: It is okay for me to lie (or not tell the truth) to someone if it will keep me from getting into trouble \n       with him or her.',
                      'f23':'f23: It is okay to steal something from someone who is rich and can easily replace it.',
                      'f24':'f24: It is okay to take little things from a store without paying for them because stores make so much money \n       that it won\'t hurt them.',
                      'f25':'f25: It is okay to beat people up if they hit me first.',
                      'f26':'f26: It is okay to beat people up if I do it to stand up for myself.', 
                      'g27':'g27: If your friends told you not to do something because it was wrong, would you listen to them?',
                      'g28':'g28: If your friends told you not to do something because it was against the law, would you listen to them?',
                      'g29':'g29: If your friends were getting you into trouble at home, would you still hang out with them?', 
                      'g30':'g30: If your friends were getting you into trouble at school, would you still hang out with them?', 
                      'g31':'g31: If your friends were getting you into trouble with the police, would you still hang out with them?', 
                      'h32':'h32: How many of your friends have skipped school without an excuse?', 
                      'h33':'h33: How many of your friends have stolen something?',
                      'h34':'h34: How many of your friends have attacked someone with a weapon (like a knife or a gun)?', 
                      'h35':'h35: How many of your friends have sold marijuana or other illegal drugs?', 
                      'h36':'h36: How many of your friends have used cigarettes, tobacco or alcohol or marijuana or other illegal drugs?', 
                      'h37':'h37: How many of your friends have belonged to a gang?', 
                      't38':'t38: How many people in your family think that you will join a gang?',
                      't39':'t39: How many people in your family are gang members?'}
        
    def plot_importance(self, absolute = False):
        #make bar plot of the last row of A to visualize
        index=np.zeros(39)
        for i in range (39):
            index[i]=i
        if absolute == False:
            plt.bar(index,self.RF)
        else:
            plt.bar(index,abs(self.RF))
        plt.title('{}: Entries in Last Row of A'.format(self.name)) 
        plt.ylabel('Entry Values')
        plt.xticks(np.arange(min(index), max(index)+1, 1.0),self.label.keys(), rotation='vertical')
        plt.show()
        
    def plot_change(self, absolute = False):
        index=np.arange(39)
        if absolute == False:
            plt.bar(index,self.w_domeigvec)
        else:
            plt.bar(index,abs(self.w_domeigvec))
        plt.title('{}: Dominant Eigenvector Entries'.format(self.name)) 
        plt.xticks(np.arange(min(index), max(index)+1, 1.0),self.label.keys(),rotation='vertical')
        plt.show()

    def plot_eiglog(self):
        # plot of log of eigenvalues
        plt.scatter(self.eiglog.real,self.eiglog.imag)
        plt.title('{}: Log(eigenvalues)'.format(self.name)) 
        plt.xlabel('Real(Growth)')
        plt.ylabel('Imaginary(Frequency)')
        plt.show()
        
    def plot_change_importance(self, absolute = True, text = True, color = 'rainbow_r', xlim = [-2,2], ylim = [-4,4]):
        
        font = {'size': 10} 
        plt.rc('font', **font)
        
        
        fig_size= plt.rcParams["figure.figsize"]
        fig_size[0] = 14
        fig_size[1] = 8
        
        plt.suptitle('{}: Change vs Importance'.format(self.name), size = 20)
        
        if absolute is False:
            scaled_ch = self.w_domeigvec
            scaled_im = self.RF
        else:
            scaled_ch = preprocessing.scale(abs(self.w_domeigvec))
            scaled_im = preprocessing.scale(abs(self.RF))
        
        colors = []
        c_dict = {'a':0,'b':1,'c':2,'d':3,'f':4,'g':5,'h':6,'t':7}
        for c in sorted(self.label.keys()):
            colors.append(c_dict[c[0]])
        colors = np.array(colors)
        
        # set up subplot grid
        gridspec.GridSpec(1,5)

        # large subplot
        if text is True:
            plt.subplot2grid((1,5), (0,0), colspan=2)
        
        plt.scatter(scaled_ch, scaled_im,c=colors, cmap=color, s=100)
       

        for i, txt in enumerate(sorted(self.label.keys())):
            plt.annotate(txt, (scaled_ch[i]+0.045, scaled_im[i]-0.05))
                
        if absolute is True:
            plt.xlim(xlim[0],xlim[1])
            plt.ylim(ylim[0],ylim[1])
        plt.axhline(0, color='red')
        plt.axvline(0, color='red')
        plt.xlabel('Standardized Weight of Entries in Dominant Eigenvector \n Change')
        plt.ylabel('Importance \n Standardized Weight of Entries in Last Row of A Matrix')
        
        
        if text is True:
            # small subplot
            plt.subplot2grid((1,5), (0,2), colspan = 3)
            # Remove the plot frame lines. They are unnecessary here.
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.axis('off')
            Text = '\n'.join(sorted(self.label.values()))
            plt.text(0,1,Text,horizontalalignment='left',verticalalignment='top')
        plt.savefig('{}_ch_im.png'.format(self.name))
        plt.show()