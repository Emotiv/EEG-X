import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)]

class PositionalEmbedding1D(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding1D, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def positional_encoding_2d(coords, d):
    """
    Generate sinusoidal positional embeddings for a list of (x, y) coordinates.
    
    Args:
        coords: List of (x, y) tuples representing positions.
        d: Embedding dimension (must be even).

    Returns:
        pos_embedding: (N, d) matrix of positional embeddings.
    """
    N = len(coords)
    pos_embedding = np.zeros((N, d))
    
    for idx, (x, y) in enumerate(coords):
        for k in range(d // 4):  # Half for x, half for y
            pos_embedding[idx, 4*k] = np.sin(x / (5 ** (4*k / d)))
            pos_embedding[idx, 4*k + 1] = np.cos(x / (5 ** (4*k / d)))
            pos_embedding[idx, 4*k + 2] = np.sin(y / (5 ** (4*k / d)))
            pos_embedding[idx, 4*k + 3] = np.cos(y / (5 ** (4*k / d)))
    
    return pos_embedding

class elec_location_Embedding(nn.Module):
    def __init__(self, d_model):
        super(elec_location_Embedding, self).__init__()
        coords = pd.read_csv("fm/EEG-X/notebooks/global_electrode_positions.csv")
        coords['x'] = (coords['x']+0.8) *5
        coords['y'] = (coords['y']+1)* 5
        new_coords = [(x, y) for x, y in zip(coords['x'], coords['y'])]
        pos_embedding = positional_encoding_2d(new_coords, d_model)

        MN8_electrodes_embeddings = match_MN8_electrodes(coords,pos_embedding)
        self.MN8_electrodes_embeddings = torch.tensor(MN8_electrodes_embeddings).float().cuda()

        Insight5_electrode_embeddings = match_Insight5_electrodes(coords,pos_embedding)
        self.Insight5_electrode_embeddings = torch.tensor(Insight5_electrode_embeddings).float().cuda()

        eight_electrodes_embeddings = match_8_electrodes(coords,pos_embedding)
        self.eight_electrodes_embeddings = torch.tensor(eight_electrodes_embeddings).float().cuda()

        epoc14_electrode_embeddings = match_epoc14_electrodes(coords,pos_embedding)
        self.epoc14_electrode_embeddings = torch.tensor(epoc14_electrode_embeddings).float().cuda()

        elec16_electrode_embeddings = match_16_electrodes(coords,pos_embedding)
        self.elec16_electrode_embeddings = torch.tensor(elec16_electrode_embeddings).float().cuda()

        THU19_electrode_embeddings = match_TUH19_electrodes(coords,pos_embedding)
        self.THU19_electrode_embeddings = torch.tensor(THU19_electrode_embeddings).float().cuda()

        BCI22_electrodes__embeddings = match_BCI22_electrodes(coords,pos_embedding)
        self.BCI22_electrodes__embeddings = torch.tensor(BCI22_electrodes__embeddings).float().cuda()

        EPFL_32_electrodes_embeddings = match_EPFL_32_electrodes(coords,pos_embedding)
        self.EPFL_32_electrodes_embeddings = torch.tensor(EPFL_32_electrodes_embeddings).float().cuda()

        Wang_60_electrodes_embeddings = match_60_electrodes(coords,pos_embedding)
        self.Wang_60_electrodes_embeddings = torch.tensor(Wang_60_electrodes_embeddings).float().cuda()

        Lee_62_electrodes_embeddings = match_Lee_62_electrodes(coords,pos_embedding)
        self.Lee_62_electrodes_embeddings = torch.tensor(Lee_62_electrodes_embeddings).float().cuda()

        PhysionetMI_64_electrodes_embeddings = match_PhysionetMI_64_electrodes(coords,pos_embedding)
        self.PhysionetMI_64_electrodes_embeddings = torch.tensor(PhysionetMI_64_electrodes_embeddings).float().cuda()

    def forward(self, x):
        if x.shape[1] == 2: # 2 electrodes
            embedding = self.MN8_electrodes_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 14: # 14 electrodes
            embedding = self.epoc14_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 5: # 5 electrodes
            embedding = self.Insight5_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 8: # 8 electrodes
            embedding = self.eight_electrodes_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 16: # 16 electrodes
            embedding = self.elec16_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)    
        elif x.shape[1] == 19: # 19 electrodes
            embedding = self.THU19_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 22: # 22 electrodes 
            embedding = self.BCI22_electrodes__embeddings.unsqueeze(1).repeat(1, x.size(2), 1) 
        elif x.shape[1] == 32: # 32 electrodes
            embedding = self.EPFL_32_electrodes_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 60: # 60 electrodes + 2 reference channels (M1 and M2 removed)
            embedding = self.Wang_60_electrodes_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 62: # 62 electrodes 
            embedding = self.Lee_62_electrodes_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 64: # 64 electrodes
            embedding = self.PhysionetMI_64_electrodes_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        else:
            print("this number of electrodes is not implemented yet")
            # return a default value just passed exported model
            embedding = self.epoc14_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)    

        return embedding

def match_MN8_electrodes(coords, pos_embedding):
    MN8_electrodes = ['t7', 't8']
    coords['label'] = coords['label'].str.lower()
    MN8_electrode_indices = coords[coords['label'].isin(MN8_electrodes)].index.tolist()
    MN8_electrode_embeddings = pos_embedding[MN8_electrode_indices]
    return MN8_electrode_embeddings

def match_Insight5_electrodes(coords, pos_embedding):
    Insight5_electrodes = ['af3', 't7', 'pz', 't8', 'af4']
    coords['label'] = coords['label'].str.lower()
    Insight5_electrode_indices = coords[coords['label'].isin(Insight5_electrodes)].index.tolist()
    Insight5_electrode_embeddings = pos_embedding[Insight5_electrode_indices]
    return Insight5_electrode_embeddings

def match_8_electrodes(coords, pos_embedding):
    electrodes = ['oz','o1','o2','poz','po3','po4','po7','po8']
    coords['label'] = coords['label'].str.lower()
    electrodes_indices = coords[coords['label'].isin(electrodes)].index.tolist()
    electrodes_indices = pos_embedding[electrodes_indices]
    return electrodes_indices

def match_epoc14_electrodes(coords, pos_embedding):
    epoc14_electrodes = ['af3', 'f7', 'f3', 'fc5', 't7', 'p7', 'o1', 'o2', 'p8', 't8', 'fc6', 'f4', 'f8', 'af4']
    coords['label'] = coords['label'].str.lower()
    epoc14_electrode_indices = coords[coords['label'].isin(epoc14_electrodes)].index.tolist()
    epoc14_electrode_embeddings = pos_embedding[epoc14_electrode_indices]
    return epoc14_electrode_embeddings

def match_16_electrodes(coords, pos_embedding):
    electrodes = ['fp1','fp2','f3','afz','f4','t7','cz','t8','p7','p3','pz','p4','p8','o1','oz','o2']
    coords['label'] = coords['label'].str.lower()
    electrodes_indices = coords[coords['label'].isin(electrodes)].index.tolist()
    electrodes_indices = pos_embedding[electrodes_indices]
    return electrodes_indices

def match_TUH19_electrodes(coords, pos_embedding):
    TUH_19_electrodes = ['fp1', 'fp2', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1',
                         'o2', 'f7', 'f8', 't7', 't8', 'p6', 'p5',
                         'fz', 'cz', 'pz']
    coords['label'] = coords['label'].str.lower()
    TUH_19_electrodes_indices = coords[coords['label'].isin(TUH_19_electrodes)].index.tolist()
    TUH_19_electrodes_indices = pos_embedding[TUH_19_electrodes_indices]
    return TUH_19_electrodes_indices

def match_BCI22_electrodes(coords, pos_embedding):
    BCI22_electrodes = ['fz', 
                          'fc3', 'fc1', 'fcz', 'fc2', 'fc4',
                           'c5', 'c3', 'c1', 'cz', 'c2', 'c4', 'c6',
                             'cp3', 'cp1', 'cpz', 'cp2', 'cp4',
                               'p1', 'pz', 'p2',
                                 'poz']
    coords['label'] = coords['label'].str.lower()
    BCI22_electrodes_indices = coords[coords['label'].isin(BCI22_electrodes)].index.tolist()
    BCI22_electrodes_embeddings = pos_embedding[BCI22_electrodes_indices]
    return BCI22_electrodes_embeddings

def match_EPFL_32_electrodes(coords, pos_embedding):
    EPFL_32_electrodes = ['fp1','af3','f7','f3','fc1','fc5','t7','c3','cp1','cp5','p7','p3','pz','po3','o1','oz','o2',
    'po4','p4','p8','cp6','cp2','c4','t8','fc6','fc2','f4','f8','af4','fp2','fz','cz']
    coords['label'] = coords['label'].str.lower()
    EPFL_32_electrodes_indices = coords[coords['label'].isin(EPFL_32_electrodes)].index.tolist()
    EPFL_32_electrodes_embeddings = pos_embedding[EPFL_32_electrodes_indices]
    return EPFL_32_electrodes_embeddings

def match_60_electrodes(coords, pos_embedding):    
    electrodes = ['fp1','fpz','fp2','af3','af4','f7','f5','f3','f1','fz','f2','f4','f6','f8','ft7','fc5',
        'fc3', 'fc1', 'fcz', 'fc2', 'fc4', 'fc6', 'ft8', 't7', 'c5', 'c3', 'c1', 'cz', 'c2', 'c4',
         'c6', 't8', 'tp7', 'cp5', 'cp3', 'cp1', 'cpz', 'cp2', 'cp4', 'cp6', 'tp8', 'p7',
          'p5', 'p3', 'p1', 'pz', 'p2', 'p4', 'p6', 'p8', 'po7', 'po5', 'po3', 'poz', 'po4', 'po6', 'po8',
           'o1', 'oz', 'o2' ]
    coords['label'] = coords['label'].str.lower()
    electrodes_indices = coords[coords['label'].isin(electrodes)].index.tolist()
    electrodes_indices = pos_embedding[electrodes_indices]
    return electrodes_indices

def match_Lee_62_electrodes(coords, pos_embedding):
    Lee_62_electrodes = ['fp1', 'fp2', 'f7','f3','fz','f4','f8','fc5','fc1','fc2','fc6','t7','c3','cz','c4','t8','tp9','cp5','cp1','cp2','cp6','tp10',
                            'p7','p3','pz','p4','p8','po9','o1','oz','o2','po10','fc3','fc4','c5','c1','c2','c6','cp3','cpz','cp4','p1','p2','poz','ft9',
                            'ftt9h','ttp7h','tp7','tpp9h','ft10','ftt10h','tpp8h','tp8','tpp10h','f9','f10','af7','af3','af4','af8','po3','po4' ]
    coords['label'] = coords['label'].str.lower()
    electrodes_indices = coords[coords['label'].isin(Lee_62_electrodes)].index.tolist()
    electrodes_indices = pos_embedding[electrodes_indices]
    return electrodes_indices

def match_PhysionetMI_64_electrodes(coords, pos_embedding):
    PhysionetMI_64_electrodes = ['fc5','fc3','fc1','fcz','fc2','fc4','fc6','c5','c3','c1','cz','c2','c4','c6','cp5','cp3','cp1',
                                'cpz','cp2','cp4','cp6','fp1','fpz','fp2','af7','af3','afz','af4','af8','f7','f5','f3','f1','fz','f2',
                                'f4','f6','f8','ft7','ft8','t7','t8','t9','t10','tp7','tp8','p7','p5','p3','p1','pz','p2','p4','p6',
                                'p8','po7','po3','poz','po4','po8','o1','oz','o2','iz']
    coords['label'] = coords['label'].str.lower()
    PhysionetMI_64_electrodes_indices = coords[coords['label'].isin(PhysionetMI_64_electrodes)].index.tolist()
    PhysionetMI_64_electrodes_embeddings = pos_embedding[PhysionetMI_64_electrodes_indices]
    return PhysionetMI_64_electrodes_embeddings


