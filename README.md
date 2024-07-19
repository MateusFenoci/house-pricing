
# House Prices: Advanced Regression Techniques

Este repositório contém o código para a competição "House Prices: Advanced Regression Techniques" do Kaggle. O objetivo da competição é prever o preço de venda de casas com base em várias características.

## Descrição do Projeto

O projeto utiliza técnicas de aprendizado de máquina para prever os preços das casas. O fluxo de trabalho do projeto inclui:

1. **Carregamento dos Dados**: Leitura dos dados de treinamento e teste fornecidos pela competição.
2. **Exploração e Limpeza dos Dados**: Identificação e tratamento de valores ausentes, transformação de variáveis categóricas em numéricas e tratamento da assimetria (skewness) nas variáveis numéricas.
3. **Preparação dos Dados**: Normalização dos dados para garantir que todas as características estejam na mesma escala.
4. **Treinamento do Modelo**: Treinamento de um modelo de Regressão com RandomForest.
5. **Avaliação do Modelo**: Uso de validação cruzada para avaliar o desempenho do modelo.
6. **Previsões e Geração de Submissão**: Geração de previsões no conjunto de dados de teste e criação do arquivo de submissão para o Kaggle.

## Estrutura do Repositório


- `house_pricing.py`: Script principal que contém todo o código para o fluxo de trabalho descrito acima.
- `train.csv`: Conjunto de dados de treinamento fornecido pela competição (não incluído no repositório).
- `test.csv`: Conjunto de dados de teste fornecido pela competição (não incluído no repositório).
- `submission.csv`: Arquivo de submissão gerado após a execução do script (não incluído no repositório).
- `README.md`: Este arquivo.
- `LICENSE.md`: Arquivo de licença do projeto.
- `.gitignore`: Arquivo que especifica quais arquivos ou diretórios devem ser ignorados pelo Git.

## Dependências

Para executar o script, você precisará das seguintes bibliotecas:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

Você pode instalar todas as dependências usando o seguinte comando:

```bash
pip install -r requirements.txt
```

## Instruções de Execução

1. Clone o repositório:

```bash
git clone https://github.com/MateusFenoci/house-pricing.git
cd house-pricing
```

2. Coloque os arquivos `train.csv` e `test.csv` na raiz do repositório.

3. Execute o script principal:

```bash
python house_pricing.py
```

4. O arquivo `submission.csv` será gerado na raiz do repositório, pronto para ser submetido no Kaggle.

## Autor

Mateus Fenoci

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE.md para mais detalhes.
