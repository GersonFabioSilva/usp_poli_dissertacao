import torch
import time
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader
from imodelsx.kan.kan_modules import KANModule, KANGAMModule
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets

from typing import List

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

if torch.cuda.is_available():
  device = torch.device("cpu")
else:
  device = torch.device("cpu")

def count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def reset_parameters(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()

def display_parameters(model, num_params=30):
    count = 0
    for name, param in model.named_parameters():
        if count >= num_params:
            break
        print(f"{name}: {param.data[:1]}")  # Mostra os primeiros 5 valores de cada parâmetro para brevidade
        count += 1

class KAN(BaseEstimator):
    def __init__(self,
                 hidden_layer_size: int = 64,
                 hidden_layer_sizes: List[int] = None,
                 regularize_activation: float = 1.0, regularize_entropy: float = 1.0, regularize_ridge: float = 0.0,
                 test_size=0.2, random_state=42, shuffle=True,
                 device: str = 'cpu',
                 **kwargs):
        '''
        Params
        ------
        hidden_layer_size : int
            If int, number of neurons in the hidden layer (assumes single hidden layer)
        hidden_layer_sizes: List with length (n_layers - 2)
            The ith element represents the number of neurons in the ith hidden layer.
            If this is passed, will override hidden_layer_size
            e.g. [32, 64] would have a layer with 32 hidden units followed by a layer with 64 hidden units
            (input and output shape are inferred by the data passed)
        regularize_activation: float
            Activation regularization parameter
        regularize_entropy: float
            Entropy regularization parameter
        regularize_ridge: float
            Ridge regularization parameter (only applies to KANGAM)
        kwargs can be any of these more detailed KAN parameters
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        '''
        if hidden_layer_sizes is not None:
            self.hidden_layer_sizes = hidden_layer_sizes
        else:
            self.hidden_layer_sizes = [hidden_layer_size]
            self.device = device
            self.regularize_activation = regularize_activation
            self.regularize_entropy = regularize_entropy
            self.regularize_ridge = regularize_ridge
            self.kwargs = kwargs   

    def fit(self, X, y, epochs = 50, early_stop = True, batch_size=512, lr=1e-3, weight_decay=1e-4, gamma=0.8):
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)
            num_outputs = len(self.classes_)
            y = torch.tensor(y, dtype=torch.long)
        else:
            num_outputs = 1
            y = torch.tensor(y, dtype=torch.float32)
        X = torch.tensor(X, dtype=torch.float32)
        num_features = X.shape[1]

        if isinstance(self, (KANGAMClassifier, KANGAMRegressor)):
            self.model = KANGAMModule(
                num_features=num_features,
                layers_hidden=self.hidden_layer_sizes,
                n_classes=num_outputs,
                **self.kwargs
            ).to(self.device)
        else:
            self.model = KANModule(layers_hidden=[num_features] +
                self.hidden_layer_sizes + [num_outputs],
            ).to(self.device)

            print(f"Número de parâmetros ajustáveis: {self.count_parameters()}")

            print("Parâmetros antes da reinicialização:")
            display_parameters(self.model)
            # Reiniciar os parâmetros antes do treinamento
            self.model.apply(reset_parameters)
            print("Parâmetros após a reinicialização:")
            display_parameters(self.model)

        X_train, X_tune, y_train, y_tune = train_test_split(X, y, test_size=batch_size, random_state=42, shuffle=True)

        dset_train = torch.utils.data.TensorDataset(X_train, y_train)
        dset_tune = torch.utils.data.TensorDataset(X_tune, y_tune)
        loader_train = DataLoader(
            dset_train, batch_size=batch_size, shuffle=True)
        loader_tune = DataLoader(
            dset_tune, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # Define loss
        if isinstance(self, ClassifierMixin):
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        tune_losses = []
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for x, labs in loader_train:
                x = x.view(-1, num_features).to(self.device)
                optimizer.zero_grad()
                output = self.model(x).squeeze()
                loss = criterion(output, labs.to(self.device).squeeze())
                if isinstance(self, (KANGAMClassifier, KANGAMRegressor)):
                    loss += self.model.regularization_loss(
                        self.regularize_activation, self.regularize_entropy, self.regularize_ridge)
                else:
                    loss += self.model.regularization_loss(
                        self.regularize_activation, self.regularize_entropy)

                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            tune_loss = 0
            with torch.no_grad():
                for x, labs in loader_tune:
                    x = x.view(-1, num_features).to(self.device)
                    output = self.model(x).squeeze()
                    tune_loss += criterion(output,
                                           labs.to(self.device).squeeze()).item()
            tune_loss /= len(loader_tune)
            tune_losses.append(tune_loss)
            scheduler.step()

            # apply early stopping
            if early_stop:
                if len(tune_losses) > 3 and tune_losses[-1] > tune_losses[-2]:
                    print("\tEarly stopping")
                    return self 
        return self

    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        output = self.model(X)
        if isinstance(self, ClassifierMixin):
            return self.classes_[output.argmax(dim=1).cpu().numpy()]
        else:
            return output.cpu().numpy()
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class KANClassifier(KAN, ClassifierMixin):
    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        output = self.model(X)
        return torch.nn.functional.softmax(output, dim=1).cpu().numpy()


class KANRegressor(KAN, RegressorMixin):
    pass


class KANGAMClassifier(KANClassifier):
    pass


class KANGAMRegressor(KANRegressor):
    pass


if __name__ == '__main__':

    df = pd.read_csv("predictive_maintenance.csv")
    df=df.drop(['Product ID','UDI','Target'],axis=1)
    label_encoder = LabelEncoder()

    # Perform label encoding
    df['Failure Type_encoded'] = label_encoder.fit_transform(df['Failure Type'])

    #df['Failure Type_encoded'].value_counts()

    df = pd.get_dummies(df, columns=['Type'], prefix='Type', drop_first=True)

    # Definição do conjunto de dados de entrada e do conjunto de dados de saída

    X_raw = df.drop(['Failure Type','Failure Type_encoded'], axis=1).values
    y_raw= df['Failure Type_encoded'].values

    #implementação do método SMOTE no conunto de dados
    smt=SMOTETomek(sampling_strategy='auto',random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X_raw, y_raw)

    # Convertendo para dataframe do pandas
    df_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i}' for i in range(X_resampled.shape[1])])
    df_resampled['Type_encoded'] = y_resampled 

    df_resampled = df_resampled.rename(columns={'feature_0':'Temp_Ar'})
    df_resampled = df_resampled.rename(columns={'feature_1':'Temp_Pr'})
    df_resampled = df_resampled.rename(columns={'feature_2':'Vel_Spindle'})
    df_resampled = df_resampled.rename(columns={'feature_3':'Torque'})
    df_resampled = df_resampled.rename(columns={'feature_4':'Desg_Ferr'})
    df_resampled = df_resampled.rename(columns={'feature_5':'Mat_L'})
    df_resampled = df_resampled.rename(columns={'feature_6':'Mat_M'})

    df_resampled = df_resampled.rename(columns={'Type_encoded':'Tipo_Falha'})
    X_ = df_resampled.drop(['Tipo_Falha'], axis="columns")
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X   = scaler.fit_transform(X_)    
    joblib.dump(scaler, 'minmax_scaler.pkl')

    y = df_resampled['Tipo_Falha']

    def grid_search(X_train, y_train, X_test, y_test, param_grid, device):

        results = []

        i = 0 

        for e_stop in param_grid['early_stop']:
            for epoch in param_grid['epochs']:
                for h_layer_size in param_grid['hidden_layer_size']:
                    for r_activation in param_grid['regularize_activation']:
                        for r_entropy in param_grid['regularize_entropy']:
                            for r_ridge in param_grid['regularize_ridge']:
                                for b_size in param_grid['batch_size']:
                                    for w_decay in param_grid['weight_decay']:
                                        for g in param_grid['gamma']:
                                            for l in param_grid['lr']:

                                                model = KANClassifier(hidden_layer_size     = h_layer_size,
                                                                    device                = device,
                                                                    regularize_activation = r_activation,
                                                                    regularize_entropy    = r_entropy,
                                                                    regularize_ridge      = r_ridge) 
                                            
                                                i+=1  

                                                
                                                progress = round(100*i/6561,2)

                                                print(f'Teste número: {i}')
                                                print(f'Progresso: {progress} %')
                                                print(f'Parâmetros: layers {h_layer_size} | activation: {r_activation} | entropy: {r_entropy} | ridge: {r_ridge} | batch size: {b_size} | decay: {w_decay} | gamma: {g} | learning reate: {l}')
                                                
                                                start_train = time.time()

                                                model.fit(X_train,
                                                          y_train,
                                                          epochs= epoch,
                                                          early_stop= e_stop,
                                                          batch_size = b_size,
                                                          lr = l,
                                                          weight_decay = w_decay,
                                                          gamma = g)
                                                
                                                end_train = time.time()

                                                y_train_pred = model.predict(X_train)
                                                f1_train = f1_score(y_train, y_train_pred, average='weighted')

                                                y_test_pred = model.predict(X_test)
                                                f1_test = f1_score(y_test, y_test_pred, average='weighted')

                                                acc = accuracy_score(y_test, y_test_pred)
                                                print(f'Acuracidade: {acc}')

                                                total_train_time = end_train-start_train

                                                result = {
                                                        'ID'                      : i,
                                                        'epochs'                  : epoch,
                                                        'early_stop'              : e_stop,
                                                        'train_time'              : total_train_time,
                                                        'accuracy'                : acc,
                                                        'f1_train'                : f1_train,
                                                        'f1_test'                 : f1_test,
                                                        'layers'                  : h_layer_size,
                                                        'regularize_activation'   : r_activation,
                                                        'regularize_entropy'      : r_entropy,
                                                        'regularize_ridge'        : r_ridge ,
                                                        'batch_size'              : b_size,
                                                        'weight_decay'            : w_decay ,
                                                        'gamma'                   : g,
                                                        'learning'                : l
                                                        }

                                                results.append(result)   
        return results
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
    'early_stop'            : [False, True],
    'epochs'                : [10, 25, 50],
    'hidden_layer_size'     : [16, 32, 64, 256, 512, 1024],
    'regularize_activation' : [0.6, 0.5, 0.4],
    'regularize_entropy'    : [0.6, 0.5, 0.4], 
    'regularize_ridge'      : [0.2, 0.1, 0.05], 
    'batch_size'            : [64, 128, 512],    
    'weight_decay'          : [0.03,0.01, 0.005],
    'gamma'                 : [0.4,0.5, 0.6],
    'lr'                    : [0.1, 0.05, 0.005]
    }

    time_start_grid = time.time()

    grid_results = grid_search(X_train, y_train, X_test, y_test, param_grid, device)

    time_end_grid = time.time()

    total_grid_search_time = time_end_grid-time_start_grid

    best_result_Accuracy        = max(grid_results, key=lambda x: x['accuracy'])
    best_result_F1_Score_train  = max(grid_results, key=lambda x: x['f1_train'])
    best_result_F1_Score_test   = max(grid_results, key=lambda x: x['f1_test'])
    best_results                = [best_result_Accuracy, best_result_F1_Score_train ,best_result_F1_Score_test]

    results_df = pd.DataFrame(grid_results)
    results_df.to_csv('grid_search_results.csv', index=False)

    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv('grid_best_results.csv', index=False)
    
    print(f'Melhor resultado por acuracia: {best_result_Accuracy}\nMelhor resultado F1 Score Train: {best_result_F1_Score_train}\nMelhor resultado F1 Score Test: {best_result_F1_Score_train}')
