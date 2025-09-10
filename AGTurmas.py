import random
import numpy as np
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


class AlocacaoTurmasAG:
    def __init__(self, turmas, salas, blocos, parametros):
        self.turmas = turmas
        self.salas = salas
        self.blocos = blocos
        self.parametros = parametros

        self.verificar_dados()
        self.preprocessar_dados()

        self.populacao = []
        self.taxa_evolucao = 0
        self.limite_insercao = 1
        self.melhor_fitness_historico = []
        self.contador_sem_evolucao = 0
        self.fitness_cache = {}

        self.salas_por_bloco = defaultdict(list)
        for sala in self.salas:
            self.salas_por_bloco[sala['bloco']].append(sala)

        self.salas_por_horario = defaultdict(list)
        for sala in self.salas:
            for horario in sala['horarios_disponiveis']:
                self.salas_por_horario[horario].append(sala)

        self._preparar_estruturas_fitness()

    def verificar_dados(self):
        ids_turmas = set()
        for turma in self.turmas:
            if turma['id'] in ids_turmas:
                raise ValueError(f"ID de turma duplicado: {turma['id']}")
            ids_turmas.add(turma['id'])

        ids_salas = set()
        for sala in self.salas:
            if sala['id'] in ids_salas:
                raise ValueError(f"ID de sala duplicado: {sala['id']}")
            ids_salas.add(sala['id'])

        turmas_especiais = [
            t for t in self.turmas if t.get('tipo') == 'especial']
        salas_especiais = [
            s for s in self.salas if s.get('tipo') == 'especial']

        if turmas_especiais and not salas_especiais:
            raise ValueError(
                "Existem turmas especiais mas nenhuma sala especial disponível")

    def _preparar_estruturas_fitness(self):
        self.horarios_por_turma = defaultdict(list)
        self.elementos_por_turma = defaultdict(list)

        for elemento in self.elementos_turma:
            self.horarios_por_turma[elemento['id_turma']].append(
                elemento['horario'])
            self.elementos_por_turma[elemento['id_turma']].append(elemento)

    def preprocessar_dados(self):
        self.elementos_turma = []
        for turma in self.turmas:
            for horario in turma['horarios']:
                elemento = {
                    'id_turma': turma['id'],
                    'horario': horario,
                    'num_alunos': turma['num_alunos'],
                    'bloco_preferencial': turma.get('bloco_preferencial', None),
                    'tipo': turma.get('tipo', 'regular'),
                    'id_elemento': f"{turma['id']}_{horario}"
                }
                self.elementos_turma.append(elemento)

        self.elementos_sala = []
        for sala in self.salas:
            for horario in sala['horarios_disponiveis']:
                elemento = {
                    'id_sala': sala['id'],
                    'horario': horario,
                    'capacidade': sala['capacidade'],
                    'bloco': sala['bloco'],
                    'tipo': sala.get('tipo', 'regular'),
                    'id_elemento': f"{sala['id']}_{horario}"
                }
                self.elementos_sala.append(elemento)

    def criar_individuo_aleatorio(self):
        individuo = {}
        salas_ocupadas_por_horario = defaultdict(set)

        for elemento in self.elementos_turma:
            horario = elemento['horario']
            salas_compativeis = [
                s for s in self.elementos_sala
                if s['horario'] == horario
                and s['tipo'] == elemento['tipo']
                and s['capacidade'] >= elemento['num_alunos']
                and s['id_elemento'] not in salas_ocupadas_por_horario[horario]
            ]

            if salas_compativeis:
                salas_preferenciais = [s for s in salas_compativeis
                                       if s['bloco'] == elemento['bloco_preferencial']]

                sala_escolhida = random.choice(
                    salas_preferenciais if salas_preferenciais else salas_compativeis)
                individuo[elemento['id_elemento']
                          ] = sala_escolhida['id_elemento']
                salas_ocupadas_por_horario[horario].add(
                    sala_escolhida['id_elemento'])

        for elemento in self.elementos_turma:
            if elemento['id_elemento'] not in individuo:
                horario = elemento['horario']
                salas_compativeis = [
                    s for s in self.elementos_sala
                    if s['horario'] == horario
                    and (s['tipo'] == elemento['tipo'] or
                         (s['tipo'] == 'regular' and elemento['tipo'] == 'regular'))
                    and s['id_elemento'] not in salas_ocupadas_por_horario[horario]
                    and s['capacidade'] >= elemento['num_alunos']
                ]

                if salas_compativeis:
                    sala_escolhida = random.choice(salas_compativeis)
                    individuo[elemento['id_elemento']
                              ] = sala_escolhida['id_elemento']
                    salas_ocupadas_por_horario[horario].add(
                        sala_escolhida['id_elemento'])
                else:
                    individuo[elemento['id_elemento']] = None

        return individuo

    def inicializar_populacao(self):
        for _ in range(self.parametros['tamanho_populacao']):
            individuo = self.criar_individuo_aleatorio()
            self.populacao.append(individuo)

    def calcular_fitness(self, individuo, use_cache=True):
        cache_key = str(individuo)
        if use_cache and cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]

        fitness_total = 0
        problemas_tamanho = 0
        problemas_bloco = 0

        for elemento_turma in self.elementos_turma:
            id_elemento = elemento_turma['id_elemento']
            if id_elemento not in individuo or individuo[id_elemento] is None:
                fitness_total += 10000
                continue

            id_sala_elemento = individuo[id_elemento]
            sala = next(
                (s for s in self.elementos_sala if s['id_elemento'] == id_sala_elemento), None)

            if not sala:
                fitness_total += 10000
                continue

            T = elemento_turma['num_alunos']
            C = sala['capacidade']

            if T > C:
                fitness_total += 10 ** (T - C)

            if C > T:
                fitness_total += (C / 30) ** (abs(C - T) / 15)
                problemas_tamanho += 1

            bloco_pref = elemento_turma['bloco_preferencial']
            if bloco_pref:
                pos_bloco_pref = self.blocos.get(
                    bloco_pref, {}).get('posicao', 0)
                pos_bloco_sala = self.blocos.get(
                    sala['bloco'], {}).get('posicao', 0)
                distancia = abs(pos_bloco_pref - pos_bloco_sala)
                fitness_total += distancia * 2  # Penalidade por distância
                if bloco_pref != sala['bloco']:
                    problemas_bloco += 1

            # Mesma sala para diferentes horários
            salas_turma = []
            for elemento in self.elementos_por_turma[elemento_turma['id_turma']]:
                if elemento['id_elemento'] in individuo and individuo[elemento['id_elemento']]:
                    sala_id = individuo[elemento['id_elemento']].split('_')[0]
                    salas_turma.append(sala_id)

            fitness_total += len(set(salas_turma))

        resultado = {
            'fitness_total': fitness_total,
            'problemas_tamanho': problemas_tamanho,
            'problemas_bloco': problemas_bloco
        }

        self.fitness_cache[cache_key] = resultado
        return resultado

    def selecionar_pais(self):
        torneio = [(ind, self.calcular_fitness(ind)['fitness_total'])
                   for ind in self.populacao]
        #torneio_ordenado = sorted(torneio_com_fitness, key=lambda x: x[1]) #Escolhe pais com melhor fitness, no artigo seleciona aleatório
        return torneio[0][0], torneio[1][0]

    
    def calcular_fitness_elemento_pai(self, pai, elemento_id):
        if elemento_id not in pai or not pai[elemento_id]:
            return float('inf')
            
        elemento_turma = next((t for t in self.elementos_turma if t['id_elemento'] == elemento_id), None)
        sala = next((s for s in self.elementos_sala if s['id_elemento'] == pai[elemento_id]), None)
        
        if not elemento_turma or not sala:
            return float('inf')
            
        return self.calcular_fitness_elemento(
            elemento_turma['num_alunos'],
            sala['capacidade'],
            elemento_turma,
            sala
        )

    def crossover(self, pai1, pai2):
        filho = {}
        salas_alocadas_por_horario = defaultdict(set)

        elementos_restantes = [e['id_elemento'] for e in self.elementos_turma
                               if e['id_elemento'] not in filho]

        taxa = min(self.parametros['taxa_crossover'], len(elementos_restantes))

        for elemento_id in random.sample(elementos_restantes, taxa):
            horario_elemento = next(e['horario'] for e in self.elementos_turma
                                    if e['id_elemento'] == elemento_id)

            pai1_valido = (elemento_id in pai1 and pai1[elemento_id] and
                           pai1[elemento_id] not in salas_alocadas_por_horario[horario_elemento])

            pai2_valido = (elemento_id in pai2 and pai2[elemento_id] and
                           pai2[elemento_id] not in salas_alocadas_por_horario[horario_elemento])

            if pai1_valido and pai2_valido:
                fitness_pai1 = self.calcular_fitness_elemento_pai(
                    pai1, elemento_id)
                fitness_pai2 = self.calcular_fitness_elemento_pai(
                    pai2, elemento_id)

                if fitness_pai1 <= fitness_pai2:
                    filho[elemento_id] = pai1[elemento_id]
                    salas_alocadas_por_horario[horario_elemento].add(
                        pai1[elemento_id])
                else:
                    filho[elemento_id] = pai2[elemento_id]
                    salas_alocadas_por_horario[horario_elemento].add(
                        pai2[elemento_id])
            else:
                filho[elemento_id] = None

        return filho

    def calcular_fitness_elemento(self, T, C, elemento_turma, sala):
        fitness = 0

        if T > C:
            fitness += 10 ** (T - C)

        if C > T:
            fitness += (C / 30) ** (abs(C - T) / 15)

        if elemento_turma['bloco_preferencial']:
            pos_bloco_pref = self.blocos.get(
                elemento_turma['bloco_preferencial'], {}).get('posicao', 0)
            pos_bloco_sala = self.blocos.get(
                sala['bloco'], {}).get('posicao', 0)
            fitness += abs(pos_bloco_pref - pos_bloco_sala) * 2

        return fitness

    def mutacao(self, individuo):
        turmas_nao_alocadas = [
            elemento for elemento in self.elementos_turma
            if individuo.get(elemento['id_elemento']) is None
        ]

        for elemento_turma in turmas_nao_alocadas:
            sala_alocada = None
            criterios = [
                self._mutacao_criterio_1,
                self._mutacao_criterio_2,
                self._mutacao_criterio_3,
                self._mutacao_criterio_4,
                self._mutacao_criterio_5
            ]

            for criterio in criterios:
                sala_alocada = criterio(individuo, elemento_turma)
                if sala_alocada:
                    individuo[elemento_turma['id_elemento']
                              ] = sala_alocada['id_elemento']
                    break

        return individuo

    def _mutacao_criterio_1(self, individuo, elemento_turma):
        for elemento in self.elementos_por_turma[elemento_turma['id_turma']]:
            if elemento['id_elemento'] in individuo and individuo[elemento['id_elemento']]:
                sala_outra = next((s for s in self.elementos_sala
                                   if s['id_elemento'] == individuo[elemento['id_elemento']]), None)
                if sala_outra:
                    id_sala_elemento = f"{sala_outra['id_sala']}_{elemento_turma['horario']}"
                    sala = next((s for s in self.elementos_sala
                                 if s['id_elemento'] == id_sala_elemento), None)
                    if (sala and sala['tipo'] == elemento_turma['tipo'] and
                            self.verificar_disponibilidade_sala(individuo, sala['id_elemento'])):
                        return sala
        return None

    def _mutacao_criterio_2(self, individuo, elemento_turma):
        if not elemento_turma['bloco_preferencial']:
            return None

        salas = [s for s in self.elementos_sala
                 if s['bloco'] == elemento_turma['bloco_preferencial'] and
                 s['horario'] == elemento_turma['horario'] and
                 s['tipo'] == elemento_turma['tipo'] and
                 s['capacidade'] >= elemento_turma['num_alunos'] and
                 s['capacidade'] <= 1.15 * elemento_turma['num_alunos'] and
                 self.verificar_disponibilidade_sala(individuo, s['id_elemento'])]

        return random.choice(salas) if salas else None

    def _mutacao_criterio_3(self, individuo, elemento_turma):
        salas = [s for s in self.elementos_sala
                 if s['horario'] == elemento_turma['horario'] and
                 s['tipo'] == elemento_turma['tipo'] and
                 s['capacidade'] >= elemento_turma['num_alunos'] and
                 s['capacidade'] <= 1.35 * elemento_turma['num_alunos'] and
                 self.verificar_disponibilidade_sala(individuo, s['id_elemento'])]

        return random.choice(salas) if salas else None

    def _mutacao_criterio_4(self, individuo, elemento_turma):
        if not elemento_turma['bloco_preferencial']:
            return None

        salas = [s for s in self.elementos_sala
                 if s['bloco'] == elemento_turma['bloco_preferencial'] and
                 s['horario'] == elemento_turma['horario'] and
                 s['tipo'] == elemento_turma['tipo'] and
                 s['capacidade'] >= elemento_turma['num_alunos'] and
                 self.verificar_disponibilidade_sala(individuo, s['id_elemento'])]

        return random.choice(salas) if salas else None

    def _mutacao_criterio_5(self, individuo, elemento_turma):
        salas = [s for s in self.elementos_sala
                 if s['horario'] == elemento_turma['horario'] and
                 s['tipo'] == elemento_turma['tipo'] and
                 s['capacidade'] >= elemento_turma['num_alunos'] and
                 self.verificar_disponibilidade_sala(individuo, s['id_elemento'])]

        return random.choice(salas) if salas else None

    def verificar_disponibilidade_sala(self, individuo, id_sala_elemento):
        return not any(value == id_sala_elemento for value in individuo.values())

    def atualizar_taxa_evolucao(self):
        melhor_individuo = min(
            self.populacao, key=lambda x: self.calcular_fitness(x)['fitness_total'])

        elementos_validos = 0
        for elemento_turma in self.elementos_turma:
            id_elemento = elemento_turma['id_elemento']
            if id_elemento not in melhor_individuo or melhor_individuo[id_elemento] is None:
                continue

            id_sala_elemento = melhor_individuo[id_elemento]
            sala = next(
                (s for s in self.elementos_sala if s['id_elemento'] == id_sala_elemento), None)

            if sala:
                fitness_elemento = self.calcular_fitness_elemento(
                    elemento_turma['num_alunos'],
                    sala['capacidade'],
                    elemento_turma,
                    sala
                )

                if fitness_elemento <= self.limite_insercao:
                    elementos_validos += 1

        total_elementos = len(self.elementos_turma)
        nova_taxa = (elementos_validos / total_elementos) * 100

        if nova_taxa <= self.taxa_evolucao:
            self.contador_sem_evolucao += 1
        else:
            self.taxa_evolucao = nova_taxa
            self.contador_sem_evolucao = max(0, self.contador_sem_evolucao - 1)

        if self.contador_sem_evolucao >= self.parametros['periodos_sem_evolucao']:
            self.limite_insercao += max(1, int(self.contador_sem_evolucao / 2))
            self.contador_sem_evolucao = 0

    def substituir_pior_individuo(self, novo_individuo):
        if len(self.populacao) < self.parametros['tamanho_populacao']:
            self.populacao.append(novo_individuo)
            return

        candidatos = self.populacao + [novo_individuo]
        candidatos_com_fitness = [(ind, self.calcular_fitness(
            ind)['fitness_total']) for ind in candidatos]

        candidatos_ordenados = sorted(
            candidatos_com_fitness, key=lambda x: x[1])
        self.populacao = [
            ind for ind, _ in candidatos_ordenados[:self.parametros['tamanho_populacao']]]

    def executar(self):
        self.verificar_dados()

        self.inicializar_populacao()
        self.contador_sem_evolucao = 0
        self.melhor_fitness_historico = []

        melhor_individuo = min(
            self.populacao, key=lambda x: self.calcular_fitness(x)['fitness_total'])
        melhor_fitness = self.calcular_fitness(
            melhor_individuo)['fitness_total']
        self.melhor_fitness_historico.append(melhor_fitness)

        print("\nIniciando algoritmo genético com as seguintes configurações:")
        print(
            f"- População: {self.parametros['tamanho_populacao']} indivíduos")
        print(f"- Gerações: {self.parametros['num_geracoes']}")
        print(f"- Fitness inicial: {melhor_fitness:.2f}")

        inicio = time.time()
        for geracao in range(self.parametros['num_geracoes']):
            try:
                for _ in range(self.parametros['novos_individuos_por_geracao']):
                    pai1, pai2 = self.selecionar_pais()
                    filho = self.crossover(pai1, pai2)
                    filho = self.mutacao(filho)
                    self.substituir_pior_individuo(filho)

                self.atualizar_taxa_evolucao()

                melhor_individuo = min(
                    self.populacao, key=lambda x: self.calcular_fitness(x)['fitness_total'])
                melhor_fitness = self.calcular_fitness(
                    melhor_individuo)['fitness_total']
                self.melhor_fitness_historico.append(melhor_fitness)

                if geracao % 10 == 0 or geracao == self.parametros['num_geracoes']:
                    print(f"\nGeração {geracao}:")
                    print(f"- Melhor fitness: {melhor_fitness:.2f}")
                    print(f"- Taxa de evolução: {self.taxa_evolucao:.2f}%")
                    print(f"- Limite de inserção: {self.limite_insercao}")

                    fit = self.calcular_fitness(melhor_individuo)
                    print(f"- Turmas não alocadas: {fit['problemas_tamanho']}")
                    print(f"- Problemas de bloco: {fit['problemas_bloco']}")

            except Exception as e:
                print(f"\nErro na geração {geracao}: {str(e)}")
                break

        tempo_total = (time.time() - inicio) / 60
        melhor_individuo = min(
            self.populacao, key=lambda x: self.calcular_fitness(x)['fitness_total'])
        resultado = self.calcular_fitness(melhor_individuo)

        print("\n=== RESULTADOS FINAIS ===")
        print(f"Melhor fitness: {resultado['fitness_total']:.2f}")
        print(f"Problemas de capacidade: {resultado['problemas_tamanho']}")
        print(
            f"Problemas de bloco preferencial: {resultado['problemas_bloco']}")
        print(f"Taxa de evolução final: {self.taxa_evolucao:.2f}%")
        print(f"Tempo de execução: {tempo_total:.2f} minutos")

        nao_alocadas = [e['id_elemento'] for e in self.elementos_turma
                        if e['id_elemento'] not in melhor_individuo or melhor_individuo[e['id_elemento']] is None]

        print(
            f"\nHorários não alocados: {len(nao_alocadas)}/{len(self.elementos_turma)}")

        self.plotar_evolucao()

        return melhor_individuo, resultado

    def plotar_evolucao(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.melhor_fitness_historico)
        plt.title("Evolução do Melhor Fitness")
        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    random.seed(27)

    blocos = {
        'PA': {'posicao': 1}, 'PB': {'posicao': 2}, 'PC': {'posicao': 3},
        'PD': {'posicao': 4}, 'PE': {'posicao': 5}, 'PF': {'posicao': 6},
        'PG': {'posicao': 7}, 'PH': {'posicao': 8}, 'PI': {'posicao': 9},
        'PJ': {'posicao': 10}, 'PK': {'posicao': 11}
    }

    salas = []
    for i in range(1, 59):
        bloco = random.choice(list(blocos.keys()))
        capacidade = random.choice(range(60, 110))
        tipo = 'regular' if random.random() > 0.1 else 'especial'

        horarios = []
        for dia in ['seg', 'ter', 'qua', 'qui', 'sex', 'sab']:
            for periodo in ['7:30', '9:30', '11:30', '13:30', '15:30', '17:30', '19:30', '21:30', '23:30']:
                horarios.append(f"{dia}_{periodo}")

        salas.append({
            'id': f"sala_{i}",
            'bloco': bloco,
            'capacidade': capacidade,
            'tipo': tipo,
            'horarios_disponiveis': horarios
        })

    turmas = []
    for i in range(1, 1349):
        num_alunos = random.choice(range(10, 110))
        bloco_pref = random.choice(list(blocos.keys()) + [None])
        tipo = 'regular' if random.random() > 0.05 else 'especial'

        num_horarios = random.randint(1, 3)
        horarios = []
        for _ in range(num_horarios):
            dia = random.choice(['seg', 'ter', 'qua', 'qui', 'sex', 'sab'])
            periodo = random.choice(
                ['7:30', '9:30', '11:30', '13:30', '15:30', '17:30', '19:30', '21:30', '23:30'])
            horarios.append(f"{dia}_{periodo}")

        turmas.append({
            'id': f"turma_{i}",
            'num_alunos': num_alunos,
            'bloco_preferencial': bloco_pref,
            'tipo': tipo,
            'horarios': horarios
        })

    parametros_otimizados = {
        'tamanho_populacao': 100,
        'num_geracoes': 200,
        'novos_individuos_por_geracao': 3,
        'taxa_crossover': 150,
        'periodos_sem_evolucao': 9
    }

    ag = AlocacaoTurmasAG(turmas, salas, blocos, parametros_otimizados)
    melhor_indiv, resultado = ag.executar()
