import numpy as np
import networkx as nx
import matplotlib as plt
from scipy.linalg import solve_triangular


def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
        line = f.readline()
        i = int(line.split()[0]) - 1
        j = int(line.split()[1]) - 1
        W[j,i] = 1.0
    f.close()

    return W


def dibujarGrafo(W, print_ejes=True):

    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}

    N = W.shape[0]
    G = nx.DiGraph(W.T)

    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])

    nx.draw(G, pos=nx.spring_layout(G), **options)

# Esta función crea la matriz diagonal D. Toma la matriz W creada anteriormente y calcula los cj haciendo la sumatoria de links.
#Finalmente redefine la diagonal como 1/cj, y así nos queda la matriz deseada.
def matrizD(W):
    D = np.zeros((len(W),len(W)))
    for i in range(len(W)):
        cj = 0
        for j in range(len(W)):
           cj += W[j][i]
        if cj != 0:
            D[i][i] = 1/cj
        else:
            D[i][i]= 0
    return D
#En esta funcion, ingresa una matriz A cuadrada y retorna una matriz triangular inferior
#y otra triangular superior que representan la factorizacion LU de la misma.
def factorizacionLU(A):
    n = len(A)
    L = np.eye(n)  # Crea la matriz L como una matriz identidad
    U = np.copy(A)  # Crea U como una copia de A
    for k in range(n - 1):  # Itera hasta la penúltima fila
        for i in range(k + 1, n):  # Itera sobre las filas debajo de la diagonal
            factor = U[i][k] / U[k][k]
            L[i][k] = factor  # Almacena factor en la matriz L
            for j in range(k, n):  # Itera sobre las columnas
                U[i][j] -= factor * U[k][j]  # Actualiza la fila i de U
    return L, U

#En esta funcion ingresa una matriz de conectividad W y un p a eleccion,
#y retorna un score y un ranking de las páginas representadas en aquella matriz W

def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind{k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha
    # Definimos cada matriz necesitada para crear la matriz (I - p W D). En este caso, se toma M como la matriz W.
    D = matrizD(M)
    I = np.eye(npages)
    A = (I- p*(M @ D))
    L , U = factorizacionLU(A)
    #Una vez calculada la factorización LU resolvemos el sistema para x utilizando que Ly = b y Ux = y
    b= np.ones(npages)
    y= solve_triangular(L,b,lower = True)
    x= solve_triangular(U,y)
    #Normalizamos el resultado
    normax= np.linalg.norm(x,1)
    scr = x/normax
    rnk = np.argsort(scr)
    return rnk, scr


def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)

    return output





