import gymnasium as gym                                     # Simula o ambiente e a interação agente-ambiente.
import numpy as np                                          # Realiza cálculos e manipula a matriz Q.
import matplotlib.pyplot as plt                             # Cria gráficos para visualizar o progresso.
import pickle                                               # Salva e recupera o aprendizado (matriz Q).

def run(episodes, is_training=True, render=False):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))                                     # iniciar um array de 500 x 6. 
                                                                                                        # Cria a matriz Q, que armazena o valor de cada combinação de estado e ação.
                                                                                                        # Serve como a "memória" do agente sobre o que ele aprendeu.
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9                                           # alfa ou taxa de aprendizagem
    discount_factor_g = 0.9                                         # gama ou taxa de desconto. Perto de 0: mais peso/recompensa colocado no estado imediato. Perto de 1: mais sobre o estado futuro.
    epsilon = 1                                                     # 1 = Ações 100% aleatórias
    epsilon_decay_rate = 0.0001                                     # taxa de decaimento épsilon. 1/0,0001 = 10.000
    rng = np.random.default_rng()                                   # gerador de números aleatórios

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]                                      # estados: 0 a 63, 0=canto superior esquerdo,63=canto inferior direito
        terminated = False                                          # Verdadeiro quando cai no buraco ou atinge o objetivo
        truncated = False                                           # Verdadeiro quando ações > 200

        rewards = 0
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:                   # Escolha da Ação (ε-greedy).
                action = env.action_space.sample()                       # ações: 0=esquerda,1=baixo,2=direita,3=cima
            else:                                                        # Durante o treinamento, o agente escolhe uma ação: Com probabilidade epsilon, toma uma ação aleatória (explorar).Caso contrário, escolhe a melhor ação conhecida com base na matriz Q (explorar o que já sabe).
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            rewards += reward

            if is_training:                                                 
                q[state,action] = q[state,action] + learning_rate_a * (                   # Atualização da Matriz Q.
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action] #Fórmula principal do Q-Learning
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)  #Após cada episódio, o epsilon diminui, reduzindo gradualmente as ações aleatórias.
                                                        #Isso força o agente a focar nas melhores ações aprendidas ao longo do tempo.

        if(epsilon==0):
            learning_rate_a = 0.0001


        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)                                        # Cria um array. Usado para armazenar a soma das recompensas de cada episódio.
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)                                                   # Plotagem do Gráfico.
    plt.savefig('taxi.png')                             

    if is_training:
        f = open("taxi.pkl","wb")                        #Salva a matriz Q treinada em um arquivo para reutilização.
        pickle.dump(q, f)                                #Isso permite continuar de onde parou, evitando treinar do zero novamente.
        f.close()

if __name__ == '__main__':
    run(1000)

    run(10, is_training=False, render=True)             #Após o treinamento, o agente utiliza a matriz Q aprendida para decidir as ações sem explorar.
                                                        #Com render=True, o ambiente é exibido graficamente, mostrando como o táxi transporta os passageiros.


























# Parte mais importante: A atualização da matriz Q e a escolha da ação (ε-greedy) são o coração do Q-Learning. Elas fazem o agente aprender quais ações levam às maiores recompensas ao longo do tempo.










# O código apresentado descreve o processo para criar um agente que aprende a resolver o problema do Taxi-V3 utilizando o algoritmo Q-Learning. 
# Ele atende aos objetivos do projeto ao incluir as seguintes funcionalidades:

                                # Q-Learning é implementado com a fórmula:

# Q(s,a) <-- Q(s,a) + a . (r+y.maxQ/a(s`,a`)-Q(s,a))
# O código aplica essa atualização em:--> 41 á 44.
# O agente aprende através da interação com o ambiente Taxi-v3 fornecido pelo Gymnasium, utilizando ações para explorar estados e acumular recompensas.


                                # Otimizar os parâmetros:

# O código oferece ajustes para os principais parâmetros que afetam o aprendizado do agente:

# α (alpha, taxa de aprendizado): Controla o quão rapidamente o Q-Table é atualizado.
# Inicialmente definido como 0.9 e ajustado para valores menores quando ε (epsilon) atinge 0, conforme observado: ----> 50, 51.

# γ (gamma, fator de desconto): Determina o peso das recompensas futuras.
# Configurado como 0.9 no código: ----> 18.

# ε (epsilon, exploração vs. exploração): Inicialmente é 1, incentivando a exploração, e diminui gradativamente com: ----> 48.


                                # Simulação e Resultados:

# O agente interage com o ambiente por 1000 episódios de treinamento, e suas recompensas são registradas em: ----> 54.
# Um gráfico é gerado para visualizar o progresso do aprendizado, mostrando o desempenho cumulativo: ----> 60 á 62.

                                # Testar o modelo treinado:

# Após o treinamento, o Q-Table (q) é salvo em um arquivo taxi.pkl. Durante os episódios de teste (is_training=False), o agente utiliza o Q-Table treinado para escolher ações, permitindo observar o comportamento aprendido:
#----> 13 E 14.

#Com o parâmetro render=True, a interação do agente com o ambiente é visualizada na interface do Gym.







































