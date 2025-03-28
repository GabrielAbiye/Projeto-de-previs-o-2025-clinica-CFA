# Previsão de Faturamento e Faltas na Clínica

## Descrição do Projeto
Este projeto tem como objetivo analisar e prever o faturamento da clínica para o ano de 2025, além de estimar a probabilidade de pacientes faltarem às consultas. A análise utiliza métodos estatísticos e de machine learning para gerar insights valiosos para a gestão da clínica.

## Objetivos
- **Prever o faturamento mensal** para o ano de 2025 utilizando modelos de séries temporais (SARIMAX);
- **Analisar padrões de ausência de pacientes** e criar um modelo preditivo para identificar prováveis faltas;
- **Gerar cenários** otimista, realista e pessimista com base no intervalo de confiança das previsões;
- **Fornecer insights para tomada de decisões** que aumentem a previsibilidade do faturamento e reduzam faltas.

## Dados Utilizados
Os dados utilizados no projeto incluem:
- Histórico de faturamento mensal;
- Registros de agendamentos, incluindo informações sobre pacientes, médicos, tipo de atendimento e forma de pagamento;
- Região de atendimento e plano de saúde dos pacientes;
- Data e horário das consultas, categorizadas por turno;
- Status do agendamento (realizado, cancelado, faltou);
- Cálculo da tabela RFV (Recência, Frequência e Valor) para segmentação dos pacientes.

## Metodologia
### Previsão de Faturamento
1. **Modelagem de Séries Temporais** com SARIMAX para prever o faturamento mensal;
2. **Geração de intervalos de confiança** para definir cenários otimista, realista e pessimista;
3. **Visualização gráfica** da projeção de faturamento para 2025.

### Previsão de Faltas
1. **Criação de um dataset preprocessado** com informações relevantes para a previsão;
2. **Treinamento de modelos de árvore de decisão** para classificação de pacientes propensos a faltar;
3. **Avaliação de desempenho** com matriz de confusão, acurácia e outras métricas;
4. **Interpretação dos resultados** e sugestões de medidas para reduzir ausências.

### Tabela RFV (Recência, Frequência e Valor)

1. **A tabela RFV** foi gerada para segmentar os pacientes com base em seu histórico de consultas, permitindo identificar perfis de comportamento. Os indicadores utilizados foram:

Recência (R): Tempo desde a última visita do paciente.

Frequência (F): Número de consultas realizadas em um período específico.

Valor (V): Faturamento gerado pelo paciente.

Com base nesses fatores, os pacientes foram categorizados para direcionar estratégias personalizadas, como campanhas de fidelização e reaquecimento da base de clientes.

## Resultados
- **Previsão de faturamento** mostrou tendência de crescimento moderado, com cenários distintos para planejamento financeiro.
- **Modelo de classificação de faltas** identificou padrões associados à ausência, permitindo intervenções direcionadas.
- **Segmentação RFV** ajudou a categorizar pacientes de acordo com sua relevância financeira para a clínica.

## Tecnologias Utilizadas
- **Python** (Pandas, NumPy, Statsmodels, Scikit-learn, Matplotlib, Seaborn)
- **Jupyter Notebook** para desenvolvimento e experimentação
- **Streamlit** para visualização interativa dos resultados (futuro desenvolvimento)

## Como Usar
1. Certifique-se de ter os pacotes necessários instalados:
   ```sh
   pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
   ```
2. Execute o notebook principal para gerar previsões e análises.
3. Visualize os resultados nos gráficos gerados.

## Próximos Passos
- Aprimorar o modelo de classificação para previsão de faltas, incluindo novas variáveis explicativas;
- Desenvolver uma interface interativa com Streamlit para acesso rápido às previsões;
- Refinar a tabela RFV para incluir insights de retenção de clientes.

---
Projeto desenvolvido por **Gabriel Abiye**.

