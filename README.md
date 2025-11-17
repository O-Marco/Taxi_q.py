# 🚕 Projeto Q-Learning no Taxi-V3

Este repositório contém uma implementação do algoritmo **Q-Learning** para treinar um agente a resolver o ambiente **Taxi-v3** do Gymnasium (anteriormente conhecido como OpenAI Gym).

O objetivo do agente é aprender a navegar em uma grade 5x5, buscar um passageiro em um dos quatro locais de embarque e deixá-lo em um dos quatro locais de desembarque designados no menor número de passos possível.

---

## 🧠 O que é Q-Learning?

**Q-Learning** é um algoritmo de Aprendizado por Reforço *Off-Policy* que permite a um agente aprender a melhor sequência de ações a tomar em um ambiente, maximizando uma recompensa total esperada.

O aprendizado é armazenado em uma **Matriz Q** (Q-Table), onde cada célula $(s, a)$ armazena o valor de tomar uma **ação** ($a$) em um determinado **estado** ($s$).

### Fórmula Central

O coração do algoritmo é a Regra de Atualização da Matriz Q. No código, a atualização é feita utilizando a seguinte fórmula:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

* $Q(s, a)$: Valor Q atual para o par estado-ação.
* $\alpha$ (Taxa de Aprendizado): Controla o quanto as novas informações substituem as antigas.
* $r$: Recompensa imediata.
* $\gamma$ (Fator de Desconto): Determina a importância das recompensas futuras.
* $\max_{a'} Q(s', a')$: O maior valor Q possível no próximo estado ($s'$).

---

## ⚙️ Detalhes da Implementação

O script `taxi_q.py` implementa as seguintes funcionalidades essenciais de um agente Q-Learning:

| Funcionalidade | Variável | Descrição |
| :--- | :--- | :--- |
| **Matriz Q** | `q` | Tabela $500 \times 6$ para armazenar o valor de cada estado-ação. |
| **Estratégia $\epsilon$-greedy** | `epsilon` | Define a probabilidade de **explorar** (ação aleatória) vs. **explorar** (melhor ação conhecida). |
| **Taxa de Aprendizado** | `learning_rate_a` ($\alpha$) | Inicialmente `0.9`. Define a rapidez com que o agente aceita novas informações. |
| **Fator de Desconto** | `discount_factor_g` ($\gamma$) | Configurado como `0.9`, valorizando recompensas futuras. |
| **Persistência** | `pickle` | A matriz Q treinada é salva em `taxi.pkl` para reutilização e teste. |

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

Certifique-se de ter o **Python 3** instalado.

### 1. Instalação das Dependências

O projeto requer os pacotes `gymnasium`, `numpy` e `matplotlib`.

```bash
pip install gymnasium[classic-control] numpy matplotlib
```
---

# 📋Treinamento e Teste
O script é configurado para realizar o treinamento e o teste automaticamente.

* Roda 1000 episódios de treinamento (run(1000)).

* Salva a matriz Q treinada em taxi.pkl.

* Roda 10 episódios de teste (run(10, is_training=False, render=True)) usando a matriz salva e exibe a visualização do táxi em ação.

## Para rodar:

```bash
python taxi_q.py
```
---
# 📊 Resultados e Progresso
Após o treinamento, o script gera o arquivo taxi.png, que ilustra o progresso do aprendizado. O gráfico exibe a soma de recompensas acumuladas em uma janela móvel dos últimos 100 episódios.

Uma curva ascendente no gráfico indica que o agente está aprendendo a resolver o ambiente de forma mais eficiente, acumulando mais recompensas de sucesso (+20 por entrega) e minimizando penalidades (-1 por movimento e -10 por ações inválidas).
