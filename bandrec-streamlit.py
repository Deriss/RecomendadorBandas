import streamlit as st

import streamlit as st

import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix
import string
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# apresentar números com 3 casas decimais
pd.set_option('display.float_format', lambda x: '%.3f' % x)

user_data = pd.read_table('./usersha1-artmbid-artname-plays_1.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])
user_profiles = pd.read_table('./usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])

artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']]
    )

user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')

popularity_threshold = 50000
user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @popularity_threshold')

combined = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')

usa_data = combined.query('country == \'United States\'')
uk_data = combined.query('country == \'United Kingdom\'')
br_data = combined.query('country == \'Brazil\'')

if not usa_data[usa_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = usa_data.shape[0]
    usa_data = usa_data.drop_duplicates(['users', 'artist-name'])
    current_rows = usa_data.shape[0]
if not uk_data[uk_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = uk_data.shape[0]
    uk_data = uk_data.drop_duplicates(['users', 'artist-name'])
    current_rows = uk_data.shape[0]
if not br_data[br_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = br_data.shape[0]
    br_data = br_data.drop_duplicates(['users', 'artist-name'])
    current_rows = br_data.shape[0]

wide_artist_data_us = usa_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_uk = uk_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_br = br_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)

# utilização do 'Compressed Sparse Row matrix'
wide_artist_data_sparse_us = csr_matrix(wide_artist_data_us.values)
wide_artist_data_sparse_uk = csr_matrix(wide_artist_data_uk.values)
wide_artist_data_sparse_br = csr_matrix(wide_artist_data_br.values)

# funções para salvar e carregar a matriz
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

#save_sparse_csr('lastfm_sparse_artist_matrix.npz', wide_artist_data_sparse)

# vamos utilizar cosine distance como nossa métrica
model_knn_us = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn_us.fit(wide_artist_data_sparse_us)
model_knn_uk = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn_uk.fit(wide_artist_data_sparse_uk)
model_knn_br = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn_br.fit(wide_artist_data_sparse_br)

def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    """
    Inputs:
    query_artist: nome do artista o qual queremos recomendações
    artist_plays_matrix: dataframe com o play count dataframe (o do pandas dataframe, não a matriz esparsa)
    knn_model: modelo que treinamos
    k: quantidade de vizinhos
    """
    # inicialização de variáveis
    query_index = None
    ratio_tuples = []
    
    
    for i in artist_plays_matrix.index:
        # faz a busca 'fuzzy' - adiciona se for parecido com a query que foi informada na entrada da função
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
   
    # apresenta resultados
    print('Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    
    # captura o índice do artista teve o melhor match 
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
    except:
        st.error('Your artist didn\'t match any artists in the data. Try again')
        return None
    
    # formatação da entrada do modelo e chamada
    vetor = np.array(artist_plays_matrix.iloc[query_index, :])
    distances, indices = knn_model.kneighbors(vetor.reshape(1, -1), n_neighbors = k)

    # apresenta os artistas selecionados 
    for i in range(0, len(distances.flatten())):
        if i == 0:
            st.subheader('Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index]))
        else:
            st.success('{0}: {1}, with distance of {2}:'.format(i, artist_plays_matrix.index[indices.flatten()[i]], distances.flatten()[i]))

    return None

# Text/Title
st.title("Recomendação de Bandas")

# Radio Buttons
pais = st.radio("Escolha o país",("Estados Unidos","Reino Unido", "Brasil"))



# Capturar input do usuário e fazer a consulta
nome = st.text_input("Digite o nome da Banda / Artista","Digite aqui..")
if st.button("Submeter"):
	if pais == "Brasil":
		data = wide_artist_data_br
		modelo = model_knn_br
	elif pais == "Reino Unido":
		data = wide_artist_data_uk
		modelo = model_knn_uk
	else:
		data = wide_artist_data_us
		modelo = model_knn_us
	print_artist_recommendations(nome, data, modelo, k = 10)
