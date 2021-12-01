from .base import BaseTrainer
import numpy as np
from numpy.linalg import inv
import copy
from tqdm import tqdm

class MABTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataloader, env, model):
        super().__init__(args, dataset, dataloader, env, model)
                
        self.n_dim = self.args.n_dim
        self.num_epochs = self.args.num_epochs
        
        if self.args.mab_type == 'linucb':
            self.mab = LinUCB(self.n_dim, self.args.num_items+1, self.args.alpha)
        elif self.args.mab_type == 'eps_greedy': 
            self.mab = EpsGreedy(self.n_dim, self.args.num_items+1, self.args.eps)
        
    @classmethod
    def code(cls):
        return 'mab'
    
    def _create_criterion(self):
        return None

    def calculate_loss(self, batch):
        return None

    def calculate_metrics(self, batch):
        return {}
    
    def train(self):
        for epoch in range(1, self.num_epochs+1):
            count = 1
            for batch in tqdm(self.train_loader):
                batch_size = next(iter(batch.values())).size(0)
                batch = {k:v.to(self.device) for k, v in batch.items()}

                B, T = batch['items'].shape

                items = batch['items'].cpu().numpy()
                ratings = batch['next_ratings'].cpu().numpy()
                arm = batch['next_items'].cpu().numpy()

                context = items[:,-self.n_dim:]
                arm = arm[:,-1]
                reward = ratings[:,-1]

                reward_arms, chosen_arms = self.mab.train(context, arm, reward, B)
                # print('Batch {} -> MAB average reward: {}'.format(count, np.mean(reward_arms)))
                # print(chosen_arms)
                count += 1
            
            # validation
            val_log_data = self.validate(mode='val')
            val_log_data['epoch'] = epoch
            self.logger.log_val(val_log_data)
            
        test_sim_data = self.simulate(mode='test')
        self.logger.log_test(test_sim_data)

    def recommend(self, state):
        items = state['items']
                
        action = []
        
        for i in range(items.shape[0]):
            context = items[i,-self.n_dim:].cpu().numpy()
            action.append(self.mab.get_action(context))
        
        return np.array(action)


class LinUCB():
    def __init__(self, ndims, narms, alpha):
        # Set number of arms
        self.narms = narms
        # Number of context features
        self.ndims = ndims
        # explore-exploit parameter
        self.alpha = alpha
        # Instantiate A as a ndims×ndims matrix for each arm
        self.A = np.zeros((self.narms, self.ndims, self.ndims))
        # Instantiate b as a 0 vector of length ndims.
        self.b = np.zeros((narms, self.ndims, 1))
        # set each A per arm as identity matrix of size ndims
        for arm in range(self.narms):
            self.A[arm] = np.eye(self.ndims)
        
        super().__init__()
        return
        
    def get_action(self, cntx):
        # gains per each arm
        p_t = np.zeros(self.narms)
        
        # MAIN LOOP 
        for i in range(self.narms):
            # initialize theta hat
            self.theta = inv(self.A[i]).dot(self.b[i])
            # get gain reward of each arm
            p_t[i] = self.theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(self.A[i]).dot(cntx)))
        action = np.random.choice(np.where(p_t==max(p_t))[0])

        return action
    
    def update(self, arm, reward, context):
        self.A[arm] = self.A[arm] + np.outer(context,context)
        self.b[arm] = np.add(self.b[arm].T, context*reward).reshape(self.ndims,1)
        return
    
    def train(self, contexts, arms, rewards, nrounds=None):
        # Initialize results 
        chosen_arms = np.zeros(nrounds)
        reward_arms = np.zeros(nrounds)
        # initialize tround and overall cumulative reward to zero
        T = 0
        G = 0

        # MAIN LOOP
        for i in range(np.shape(contexts)[0]):
            action = self.get_action(contexts[i,:])
            if T<nrounds:
                if action == arms[i]:
                    # get the reward of chosen arm at round T
                    reward_arms[T] = rewards[i]
                    # Update matrix A and b 
                    self.update(action, rewards[i], contexts[i,:])
                    # store chosen arm at round T
                    chosen_arms[T] = action
                    T +=1
            else:
                # if desired tround ends, terminate the loop
                break
        return reward_arms, chosen_arms
    
class EpsGreedy():
    def __init__(self, ndims, narms, eps):
        # Set number of arms
        self.narms = narms
        # Number of context features
        self.ndims = ndims
        # explore-exploit parameter
        self.eps = eps
        # Instantiate A as a ndims×ndims matrix for each arm
        self.A = np.zeros((self.narms, self.ndims, self.ndims))
        # Instantiate b as a 0 vector of length ndims.
        self.b = np.zeros((narms, self.ndims, 1))
        # set each A per arm as identity matrix of size ndims
        for arm in range(self.narms):
            self.A[arm] = np.eye(self.ndims)
            
        super().__init__()
        return
        
    def get_action(self, cntx):
        # gains per each arm
        p_t = np.zeros(self.narms)
        
        # Generate random number
        p = np.random.rand()
        
        if p < self.eps:
            action = np.random.choice(self.narms)
        else:
            for i in range(self.narms):
                # initialize theta hat
                self.theta = inv(self.A[i]).dot(self.b[i])
                p_t[i] = self.theta.T.dot(cntx)
            action = np.random.choice(np.where(p_t==max(p_t))[0])

        return action
    
    def update(self, arm, reward, context):
        self.A[arm] = self.A[arm] + np.outer(context,context)
        self.b[arm] = np.add(self.b[arm].T, context*reward).reshape(self.ndims,1)
        return
    
    def train(self, contexts, arms, rewards, nrounds=None):
        # Initialize results 
        chosen_arms = np.zeros(nrounds)
        reward_arms = np.zeros(nrounds)
        # initialize tround and overall cumulative reward to zero
        T = 0
        G = 0

        # MAIN LOOP
        for i in range(np.shape(contexts)[0]):
            action = self.get_action(contexts[i,:])
            if T<nrounds:
                if action == arms[i]:
                    # get the reward of chosen arm at round T
                    reward_arms[T] = rewards[i]
                    # Update matrix A and b 
                    self.update(action, rewards[i], contexts[i,:])
                    # store chosen arm at round T
                    chosen_arms[T] = action
                    T +=1
            else:
                # if desired tround ends, terminate the loop
                break
        return reward_arms, chosen_arms