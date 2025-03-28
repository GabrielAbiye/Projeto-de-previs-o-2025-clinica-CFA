import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors
import re

from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from gower import gower_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy.cluster.hierarchy import linkage , fcluster , dendrogram
from scipy.spatial.distance import pdist , squareform
from scipy.stats import pearsonr, chi2_contingency, f_oneway

from io import BytesIO


# Configuração da página
st.set_page_config(page_title='Projeto clínica CFA', layout="wide", initial_sidebar_state='expanded')

# Função para converter DataFrame para CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def categorizar_regiao(bairro):
    if bairro in ["ITAPUÃ", "NOVA BRASILIA DE ITAPUÃ", "ALTO DO COQUEIRINHO", "PLACAFORD", 
                  "FAZENDA GRANDE DO RETIRO", "ITINGA", "PIATÃ", "MUSSURUNGA", "STELLA MARIS", 
                  "BAIRRO DA PAZ", "CASSANGE" , "PARQUE SÃO CRISTÓVÃO",  "SÃO CRISTÓVÃO"]:
        return 'Região de Itapuã e São Cristóvão'
    elif bairro in ["SUSSURANA", "PAU DA LIMA", "VILA RUY BARBOSA", "PERNAMBUÉS", 
                    "JARDIM NOVA ESPERANÇA", "TROBOGY", "PITUÁÇU", "SÃO MARCOS", "TANCREDO NEVES"]:
        return 'Subúrbio e Região Norte de Salvador'
    elif bairro in ["LIBERDADE", "CAJI", "CENTRO", 
                    "CAIXA D'ÁGUA", "VIDA NOVA", "CASTELO BRANCO", 
                    "7 DE ABRIL", "SANTA CRUZ"]:
        return 'Centro de Salvador e Regiões Circunvizinhas'
    elif bairro in ["CAMAÇARI", "PORTO DE SAUIPE", "PRAIA DO FLAMENGO"]:
        return 'Litoral Norte de Salvador e Municípios Vizinhos'
    else:
        return 'Outras regiões'
    
@st.cache_data
def turno(horario):
    try:
        if pd.isna(horario):
            return None
        if horario.hour < 12:
            return "matutino"
        return "vespertino"
    except AttributeError:
        return None
    
@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

@st.cache_data
def padronizar_bairro(bairro):
    if pd.isna(bairro):
        return "NÃO INFORMADO"
    bairro = bairro.upper().strip()
    mapeamento = {
    'ITAPUÃ': 'ITAPUÃ',
    'ITAPUAN': 'ITAPUÃ',
    'IMBUI': 'IMBUÍ',
    'IMBUÍ': 'IMBUÍ',
    'VILAS DO ATLANTICO': 'VILAS DO ATLÂNTICO',
    'VILAS DO ATLÂNTICO': 'VILAS DO ATLÂNTICO',
    '7 DE ABRIL': '7 DE ABRIL',
    'SETE DE ABRIL': '7 DE ABRIL',
    'SÃO CAETANO': 'SÃO CAETANO',
    'SAO CAETANO': 'SÃO CAETANO',
    'FAROL DE ITAPUÃ': 'FAROL DE ITAPUÃ',
    'JARDIM ARMAÇÃO': 'JARDIM ARMAÇÃO',
    'CAMAÇARI': 'CAMAÇARI',
    'CAMAÇARI DE DENTRO': 'CAMAÇARI',
    'JAGUARIPE 1': 'JAGUARIBE',
    'JAGUARIPE': 'JAGUARIBE',
    'IPITANGA LAURO DE FREITAS' : 'IPITANGA',
    'ALPHAVILLE I' : 'ALPHAVILLE' ,
    'ALPHAVILE 1' : 'ALPHAVILLE',
    'PARAFUSO' : 'VIA PARAFUSO',
    'CAZEIRAS 11' : 'CAJAZEIRAS',
    'SAN MARTIN' : 'SAN MARTINS',
    'JARDIM PLACAFORD' : 'PLACAFORD'
    
    #adicionar mais caso necessário    
    }
    return mapeamento.get(bairro, bairro)

@st.cache_data
def padronizar_plano(plano):
    planos_ativos = ['PARATODOS', 'ASSEBA', 'ASTEBA', 'PLANSERV']
    

    if pd.isna(plano):
        return 'PARTICULAR'
    
    if not isinstance(plano, str):
        return 'PARTICULAR'
    
    plano = plano.upper().strip()

    if plano in planos_ativos:
        return plano
    else:
        return 'PARTICULAR'

@st.cache_data
def padronizar_status(status_col, data_col):
    status = []
    
    for status_atual, data_agendamento in zip(status_col, data_col):
        if status_atual in ['Finalizado', 'Faltou']:
            status.append(status_atual)
        elif status_atual in ['Agendado', 'Confirmado']:
            if data_agendamento < datetime.now():
                status.append('Faltou')
            else:
                status.append('Agendado')
        elif status_atual in ['Presente', 'Em atendimento']:
            status.append('Finalizado')
        elif status_atual == 'Cancelou':
            status.append('Cancelado')
        else:
            # Mantém o valor original se não se enquadrar em nenhum caso anterior
            status.append(status_atual)
    
    return status

@st.cache_data
def recencia_class(x, r, q_dict):
    """Classifica como melhor o menor quartil 
       x = valor da linha,
       r = recencia,
       q_dict = quartil dicionario   
    """
    if x <= q_dict[r][0.25]:
        return 'A'
    elif x <= q_dict[r][0.50]:
        return 'B'
    elif x <= q_dict[r][0.75]:
        return 'C'
    else:
        return 'D'

@st.cache_data
def freq_val_class(x, fv, q_dict):
    """Classifica como melhor o maior quartil 
       x = valor da linha,
       fv = frequencia ou valor,
       q_dict = quartil dicionario   
    """
    if x <= q_dict[fv][0.25]:
        return 'D'
    elif x <= q_dict[fv][0.50]:
        return 'C'
    elif x <= q_dict[fv][0.75]:
        return 'B'
    else:
        return 'A'


def main():
    st.title('PROJETO CLÍNICA CFA')
    st.markdown('---')

    st.sidebar.write("## Suba os arquivos (Pacientes, Produtividade, Repasse)")
    uploaded_files = st.sidebar.file_uploader(
    "Faça upload dos arquivos", 
    type=['csv', 'xlsx'], 
    accept_multiple_files=True
)

# Criar dicionário para armazenar os DataFrames
    dataframes = {}

    if uploaded_files:
        for file in uploaded_files:
            file_name = file.name.lower()  # Normaliza o nome do arquivo para minúsculas
            
            if "pacientes" in file_name:
                dataframes["pacientes"] = pd.read_excel(file)
                dataframes["pacientes"].columns = dataframes["pacientes"].columns.str.lower()
                
                dataframes["pacientes"]['bairro'] = dataframes["pacientes"]['bairro'].apply(padronizar_bairro)
                dataframes["pacientes"]['plano'] = dataframes["pacientes"]['plano'].apply(padronizar_plano)
                dataframes["pacientes"].drop(columns='convênio', inplace=True)

                cols_data = ['data de nascimento', 'data do último agendamento', 
                            'data do próximo agendamento', 'data de criação']
                dataframes["pacientes"][cols_data] = dataframes["pacientes"][cols_data].apply(
                    pd.to_datetime, errors='coerce', dayfirst=True
                )

                dataframes["pacientes"]['idade'] = dataframes["pacientes"]['data de nascimento'].apply(
                    lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else None)
                
            elif "produtividade" in file_name:
                dataframes["produtividade"] = pd.read_excel(file)
                dataframes["produtividade"].columns = dataframes["produtividade"].columns.str.lower()

                # Tratamento de datas
                cols_data = ['data do agendamento', 'agendado em']
                dataframes["produtividade"][cols_data] = dataframes["produtividade"][cols_data].apply(
                    pd.to_datetime, errors='coerce', dayfirst=True
                )

                # Remover idade e adicionar nova idade do pacientes
                dataframes["produtividade"].drop(columns='idade', inplace=True, errors='ignore')

                if "pacientes" in dataframes:
                    dataframes["produtividade"] = dataframes["produtividade"].merge(
                        dataframes["pacientes"][['id amigo', 'idade']], on='id amigo', how='left'
                    )

                # Padronizar status do agendamento
                dataframes["produtividade"]['status do agendamento'] = padronizar_status(
                    dataframes["produtividade"]['status do agendamento'], 
                    dataframes["produtividade"]['data do agendamento']
                )

            elif "repasse" in file_name:
                dataframes["repasse"] = pd.read_excel(file)
                dataframes["repasse"].columns = dataframes["repasse"].columns.str.lower()

                # Converter colunas de datas corretamente
                dataframes["repasse"]['data atend'] = pd.to_datetime(
                    dataframes["repasse"]['data atend'], errors='coerce', dayfirst=True
                )

                # Substituir valores na coluna 'forma de pagamento'
                dataframes["repasse"]['forma de pagamento'] = dataframes["repasse"]['forma de pagamento'].replace('Pix', 'Crédito em conta'
                )    

    arquivos_necessarios = ['pacientes', 'produtividade', 'repasse']
    
    for arquivo in arquivos_necessarios:
        if arquivo not in dataframes:
            st.write(f"**VOCÊ NÃO CARREGOU O ARQUIVO {arquivo.upper()}!**")
    # Exibir mensagem de sucesso
    for key, df in dataframes.items():
        st.write(f"{key.capitalize()} carregado e tratamentos aplicados com sucesso!")
        st.dataframe(df)

    if all(key in dataframes for key in ['pacientes', 'repasse', 'produtividade']):
            pacientes = dataframes['pacientes']
            produtividade = dataframes['produtividade']
            repasse = dataframes['repasse']

            st.subheader('Análises gráficas' , divider= 'blue')
            df_fin = pacientes[['id amigo' , 'sexo' , 'bairro' , 'plano']]
            df_fin = df_fin.merge(repasse , on = 'id amigo' , how = 'right').dropna().reset_index()
            df_fin.drop(columns = 'index' , inplace = True)
            df_fin['faturamento_clinica'] = df_fin.apply(
            lambda row: row['líquido'] if row['medico'] == 'ULTRASSONOGRAFISTA' else row['líquido'] - row['repasse executante'],
            axis=1
            )
            df_fin['regiao'] = df_fin.bairro.apply(categorizar_regiao)

            plt.figure(figsize = (20 , 8))
            sns.barplot(df_fin.bairro.value_counts().head(10))
            plt.xticks(rotation=90)
            plt.title('Bairros com maior incidencia', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()

            st.pyplot(plt)

            ano_selecionado = st.selectbox(
                "Selecione o ano para análise:",
                options=[2023, 2024],
                index=1  # Default para 2024
            )

            # Carregar os dados baseados no ano selecionado
            dados_ano = df_fin[df_fin['data atend'].dt.year == ano_selecionado].copy()
            dados_ano.loc[:, 'mes_ano'] = dados_ano['data atend'].dt.to_period('M')
            prod_mens_ano = dados_ano.groupby(['mes_ano', 'medico'])['líquido'].sum().reset_index()
            prod_mens_ano['mes_ano'] = prod_mens_ano['mes_ano'].astype(str)

            # Plotando a produção mensal para o ano selecionado
            plt.figure(figsize=(20, 8))
            sns.lineplot(data=prod_mens_ano, x='mes_ano', y='líquido', errorbar=('ci', 0))
            plt.title(f'Produção por período em {ano_selecionado}', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            
            st.pyplot(plt)

            # Lista de especialidades
            especialidades = [
                'PSICOLOGISTA', 'ULTRASSONOGRAFISTA', 'GINECOLOGISTA 1',
                'PARCEIROS EXTERNOS', 'CARDIOLOGISTA', 'ANGIOLOGISTA',
                'CLINICO GERAL', 'NEUROLOGISTA', 'NUTRICIONISTA', 'GINECOLOGISTA 2'
            ]

            # Caixa de seleção para as especialidades, usando um key único para evitar conflito
            especialidades_selecionadas = st.multiselect(
                "Selecione as especialidades para comparar:",
                especialidades,
                default=['PSICOLOGISTA', 'CARDIOLOGISTA'],  # Exemplo de valores padrão
                key=f"especialidades_{ano_selecionado}"  # Chave única por ano
            )

            # Gerar o gráfico comparativo se ao menos 2 especialidades forem selecionadas
            if len(especialidades_selecionadas) >= 2:
                filtro_comparacao = prod_mens_ano[prod_mens_ano['medico'].isin(especialidades_selecionadas)]

                # Gerando o gráfico de comparação entre especialidades
                plt.figure(figsize=(20, 8))
                sns.lineplot(data=filtro_comparacao, x='mes_ano', y='líquido', hue='medico', errorbar=('ci', 0))
                plt.title(f'Comparação de Produção entre Especialidades em {ano_selecionado}', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.tight_layout()
                
                st.pyplot(plt)
            else:
                st.write("Por favor, selecione pelo menos duas especialidades para comparar.")

            st.subheader('**TABELA RFV**' , divider = 'blue')
            st.write("""
            **RFV** significa recência, frequência, valor e é utilizado para segmentação de clientes baseado no comportamento de compras dos clientes e agrupa eles em clusters parecidos.

            Utilizando esse tipo de agrupamento podemos realizar ações de marketing e CRM melhores direcionadas, ajudando assim na personalização do conteúdo e até a retenção de clientes.

            Para cada cliente é preciso calcular cada uma das componentes abaixo:""")
            
            st.write("""
            - **Recência**: Quantos dias se passaram desde a última compra.  
            - **Frequência**: Quantidade de compras realizadas em um período.  
            - **Valor**: Total gasto pelo cliente.
            """)


            dia_atual = datetime(2025, 1, 3)
            df_recencia = df_fin.groupby(by='id amigo',
                                 as_index=False)['data atend'].max()
            df_recencia.columns = ['id amigo', 'ultimo atend']
            df_recencia['Recencia'] = df_recencia['ultimo atend'].apply(
                lambda x:(dia_atual - x).days)
            df_recencia.drop('ultimo atend' ,axis = 1, inplace = True)
            df_frequencia = df_fin.groupby('id amigo').size().reset_index(name='Frequencia')
            df_valor = df_fin[['id amigo', 'faturado'
                       ]].groupby('id amigo').sum().reset_index()
            df_valor.columns = ['id amigo', 'Valor']
            df_rf = df_recencia.merge(df_frequencia , on = 'id amigo')
            df_RFV = df_rf.merge(df_valor, on='id amigo')
            df_RFV.set_index('id amigo', inplace=True)
            st.dataframe(df_RFV)

            quartis = df_RFV.quantile(q=[0.25, 0.5, 0.75])
            quartis.to_dict()

            df_RFV['R_quartil'] = df_RFV['Recencia'].apply(recencia_class,
                                                args=('Recencia', quartis))
            df_RFV['F_quartil'] = df_RFV['Frequencia'].apply(freq_val_class,
                                                            args=('Frequencia', quartis))
            df_RFV['V_quartil'] = df_RFV['Valor'].apply(freq_val_class,
                                                        args=('Valor', quartis))
            
            df_RFV['RFV_Score'] = (df_RFV.R_quartil + df_RFV.F_quartil +
                       df_RFV.V_quartil)
            
            variaveis = df_RFV.iloc[: , :-1].columns.values
            variaveis_quant = df_RFV.iloc[: , :3].columns.values
            variaveis_cat = df_RFV.iloc[:, 3:-1].columns.values

            df_pad = pd.DataFrame(StandardScaler().fit_transform(df_RFV[variaveis_quant]) , columns = df_RFV[variaveis_quant].columns)
            df_pad[variaveis_cat] = df_RFV[variaveis_cat].values

            df_rfv_dummies = pd.get_dummies(df_pad[variaveis].dropna() , columns = variaveis_cat , dtype = int)

            colunas_categoricas = set(df_rfv_dummies.iloc[: , 3:].columns)

            vars_cat = [col in colunas_categoricas for col in df_rfv_dummies.columns]
            distancia_gower = gower_matrix(df_rfv_dummies, cat_features=vars_cat)
            gdv = squareform(distancia_gower, force='tovector')

            Z = linkage(gdv , method = 'complete')
            Z_df = pd.DataFrame(Z , columns = ['ind1' , 'ind2' , 'dist' , 'n'])

            df_RFV['5 grupos'] = fcluster(Z , 5 , criterion = 'maxclust')

            df_RFV['index'] = df_rfv_dummies.index
            df_RFV_fin = df_RFV.merge(df_rfv_dummies.reset_index() , on = 'index' , how = 'left')
            df_RFV_fin = df_RFV.reset_index().merge(df_rfv_dummies.reset_index() , on = 'index' , how = 'left')

            df_RFV.drop(columns = ['R_quartil' ,'F_quartil' , 'V_quartil' , 'index'] , inplace = True)
            plt.figure(figsize=(20, 8))
            sns.lineplot(x='5 grupos', y='RFV_Score', data=df_RFV_fin, marker='o')
            plt.title('RFV Score por grupo(5 grupos)')
            plt.xlabel('Grupo')
            plt.ylabel('RFV Scores')

            st.pyplot(plt)

            st.write("### Ações de marketing/CRM")

            dict_acoes = {
                1: 
                "Cliente fidelizado. Recompensar com cupons de desconto exclusivos e programas de indicação. Incentivar avaliações e depoimentos positivos.",
                2: 
                "Cliente com alto potencial de fidelização. Enviar mensagem de agradecimento personalizada e oferecer benefícios para próximas compras.",
                3: 
                "Cliente com baixo ticket médio. Criar promoções direcionadas, oferecer combos e incentivar compras recorrentes.",
                4: 
                "Cliente inativo. Reativar com campanhas personalizadas, ofertas exclusivas e conteúdos relevantes para despertar interesse.",
                5: 
                "Cliente com pouca interação e baixo volume de compras. Analisar suas preferências e criar estratégias para aumentar o engajamento, como ofertas personalizadas, brindes e benefícios progressivos."  
            }

            df_RFV['acoes de marketing/crm'] = df_RFV['5 grupos'].map(dict_acoes)

            st.dataframe(df_RFV)

            st.write("### Baixar a Tabela RFV em Excel")

            excel_file = to_excel(df_RFV)
            st.download_button(
                label="📥 Baixar Excel",
                data=excel_file,
                file_name="RFV.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )



            st.subheader('Previsão de renda 2025' , divider = "blue")

            df_fin.reset_index(inplace = True)
            df_fin['data atend'] = pd.to_datetime(df_fin['data atend'])

            # Agrupar por mês e calcular o faturamento total
            faturamento_mensal = df_fin.groupby(pd.Grouper(key='data atend', freq='M'))['faturamento_clinica'].sum().reset_index()

            plt.figure(figsize=(20, 8))
            plt.plot(faturamento_mensal.index, faturamento_mensal['faturamento_clinica'], label='Faturamento Mensal')
            plt.title('Faturamento Mensal da Clínica (2023-2024)')
            plt.xlabel('Data')
            plt.ylabel('Faturamento')
            plt.legend()

            st.pyplot(plt)

            decomposicao = seasonal_decompose(faturamento_mensal['faturamento_clinica'], model='additive', period=12)
            modelo = SARIMAX(
            faturamento_mensal['faturamento_clinica'],
            order=(1, 1, 1),  # (p, d, q)
            seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s) com s=12 para sazonalidade anual
            )
            resultado = modelo.fit(disp=False)
            previsao = resultado.get_forecast(steps=12)
            previsao_media = previsao.predicted_mean
            intervalo_confianca = previsao.conf_int()

            plt.figure(figsize=(20, 8))
            plt.plot(faturamento_mensal.index, faturamento_mensal['faturamento_clinica'], label='Dados Históricos')
            plt.plot(previsao_media.index, previsao_media, label='Previsão 2025', color='red')
            plt.fill_between(
                intervalo_confianca.index,
                intervalo_confianca.iloc[:, 0],
                intervalo_confianca.iloc[:, 1],
                color='pink', alpha=0.3, label='Intervalo de Confiança'
            )
            plt.title('Previsão de Faturamento para 2025')
            plt.xlabel('Data')
            plt.ylabel('Faturamento')
            plt.legend()

            st.pyplot(plt)

            df_cenarios = pd.DataFrame({
            "Data": previsao_media.index,
            "Cenário Ruim": intervalo_confianca.iloc[:, 0],  # Limite inferior
            "Cenário Normal": previsao_media,                # Previsão central
            "Cenário Ótimo": intervalo_confianca.iloc[:, 1]  # Limite superior
            })

            plt.figure(figsize=(20, 8))

            # Dados históricos
            plt.plot(faturamento_mensal.index, faturamento_mensal['faturamento_clinica'], label='Dados Históricos', color='black')

            # Cenários
            plt.plot(df_cenarios["Data"], df_cenarios["Cenário Normal"], label="Cenário Normal", color="blue")
            plt.plot(df_cenarios["Data"], df_cenarios["Cenário Ruim"], linestyle="dashed", color="red", label="Cenário Ruim")
            plt.plot(df_cenarios["Data"], df_cenarios["Cenário Ótimo"], linestyle="dashed", color="green", label="Cenário Ótimo")

            # Sombreamento do intervalo de confiança
            plt.fill_between(df_cenarios["Data"], df_cenarios["Cenário Ruim"], df_cenarios["Cenário Ótimo"], color="gray", alpha=0.2, label="Intervalo de Confiança")

            plt.title('Cenários de Previsão de Faturamento para 2025')
            plt.xlabel('Data')
            plt.ylabel('Faturamento')
            plt.legend()
            plt.xticks(rotation=45)

            plt.savefig('output/Cenarios_Faturamento_2025.png', dpi=300, bbox_inches='tight')

            st.pyplot(plt)

            meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
         "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
            df_cenarios['Data'] = meses

            st.dataframe(df_cenarios)

            st.write("### Baixar tabela Previsão 2025 em Excel")

            excel_file = to_excel(df_cenarios)
            st.download_button(
                label="📥 Baixar Excel",
                data=excel_file,
                file_name="Cenarios 2025.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )







          
    else:
        st.write("**POR FAVOR, CARREGUE TODOS OS ARQUIVOS PARA SEGUIR COM AS ANÁLISES**")


# Iniciar a aplicação
if __name__ == '__main__':
    main()
