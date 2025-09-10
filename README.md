# Aplicação de Algoritmos Genéticos e Métodos Evolucionários na Resolução do Problema de Alocação de Turmas: Caso UFPR

## Descrição do Projeto

Este projeto aborda o **problema de alocação de turmas** dos 19 cursos de graduação do campus Centro Politécnico da UFPR, distribuídas em **58 salas** localizadas em **11 blocos**, respeitando restrições de **capacidade**, **horários** e **preferências de localização**.

O problema é classificado como **NP-Hard**, sendo tratado através de **heurísticas e metaheurísticas**, especificamente utilizando **Algoritmos Genéticos (AG)**, para encontrar soluções viáveis e eficientes.

O desafio central é alocar cada turma (tarefa) em uma sala e horário (recurso), respeitando regras rígidas (capacidade da sala, não sobreposição de horários) e regras flexíveis (preferência de blocos, laboratórios específicos, etc.).

## Representação do Indivíduo

Cada indivíduo da população é representado como uma **lista de genes**, onde cada gene corresponde a uma turma e contém os seguintes atributos:

* **Turma ID** → Identificação única da turma.
* **Sala ID** → Sala onde a turma será alocada.
* **Horário** → Intervalo de tempo da aula.
* **Bloco** → Bloco do campus onde está a sala.
* **Número de Alunos** → Quantidade de alunos da turma.
* **Restrições Especiais** → Preferências ou requisitos específicos (laboratórios, ateliês, etc.).

## Funcionalidades Implementadas

O código do projeto implementa:

* **Criação de população inicial** de indivíduos aleatórios respeitando restrições de capacidade e horários.
* **Cálculo de fitness**, considerando:

  * Penalidades por excesso de alunos em uma sala.
  * Penalidades por utilização de salas acima do necessário.
  * Penalidades por não atender preferências de bloco.
  * Penalidades por salas não alocadas.
* **Seleção de pais** para crossover via torneio.
* **Operador de crossover** que combina genes de dois pais, priorizando menores penalidades.
* **Operador de mutação** baseado em critérios de realocação de turmas não alocadas.
* **Atualização dinâmica da taxa de evolução**, aumentando o limite de inserção em caso de estagnação.
* **Substituição do pior indivíduo** da população com o novo indivíduo gerado.
* **Execução do AG** com histórico de evolução do fitness e relatórios de resultados.

## Tecnologias Utilizadas

* **Python 3**
* Bibliotecas:

  * `numpy` → operações matemáticas e vetoriais.
  * `matplotlib` → plotagem da evolução do fitness.
  * `collections` → estruturas como `defaultdict` e `Counter`.
  * `random` → geração de números aleatórios.

## Estrutura do Projeto

O código principal está contido na classe `AlocacaoTurmasAG`, que inclui métodos para:

* Inicialização de dados (`preprocessar_dados`, `verificar_dados`)
* Criação e mutação de indivíduos (`criar_individuo_aleatorio`, `mutacao`)
* Seleção e crossover (`selecionar_pais`, `crossover`)
* Cálculo de fitness (`calcular_fitness`, `calcular_fitness_elemento`)
* Controle da evolução (`atualizar_taxa_evolucao`, `substituir_pior_individuo`)
* Execução do algoritmo e plotagem de resultados (`executar`, `plotar_evolucao`)

## Configuração e Execução

1. Defina os blocos e salas do campus, incluindo capacidade, horários e tipo (`regular` ou `especial`).
2. Defina as turmas, incluindo número de alunos, blocos preferenciais, tipo e horários disponíveis.
3. Configure os parâmetros do algoritmo genético:

```python
parametros_otimizados = {
    'tamanho_populacao': 100,
    'num_geracoes': 200,
    'novos_individuos_por_geracao': 3,
    'taxa_crossover': 150,
    'periodos_sem_evolucao': 9
}
```

4. Inicialize a classe `AlocacaoTurmasAG`:

```python
ag = AlocacaoTurmasAG(turmas, salas, blocos, parametros_otimizados)
melhor_indiv, resultado = ag.executar()
```

5. O algoritmo exibirá:

   * Melhor fitness alcançado
   * Número de problemas de capacidade e bloco
   * Taxa de evolução final
   * Tempo total de execução
   * Horários não alocados
   * Gráfico da evolução do melhor fitness ao longo das gerações

## Resultados

O algoritmo é capaz de gerar **soluções viáveis** para o problema de alocação de turmas, respeitando restrições rígidas e minimizando penalidades por preferências não atendidas. A visualização da evolução do fitness permite acompanhar a convergência do AG e ajustar parâmetros para melhor desempenho.

## Referências

* \[Chen, S. & Shih, C. (2013). Solving University Course Timetabling Problems Using Constriction Particle Swarm Optimization with Local Search.]
* Pesquisa aplicada ao **Problema de Alocação de Turmas (Timetabling)** no campus Centro Politécnico da UFPR.
* Implementação baseada em heurísticas e metaheurísticas em Python.

