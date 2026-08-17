# BTC Sentiment LSTM

Previsão de movimento de preços do Bitcoin (BTC-USD) combinando dados numéricos
(preço, volume, indicadores técnicos) com sentimento de notícias extraído via
**FinBERT**, avaliada com metodologia honesta de séries temporais financeiras:
baselines fortes, split *walk-forward* sem vazamento temporal, múltiplas
sementes com teste estatístico pareado, interpretabilidade e simulação de
trading com custos de transação.

> **Pergunta de pesquisa:** adicionar sentimento de notícias a um LSTM que já
> usa preço/volume/indicadores técnicos melhora, de forma estatisticamente
> significativa, a previsão do fechamento do dia seguinte ($t+1$) do BTC-USD?
>
> **Resultado:** majoritariamente nulo — de 4 métricas testadas (RMSE, MAE,
> acurácia, F1), só o F1 deu significativo em um teste (t pareado, p=0,035) e
> não no outro (Wilcoxon, p=0,056). Discutimos isso como evidência consistente
> com a Hipótese de Mercado Eficiente em sua forma semi-forte — um resultado
> nulo bem controlado é, por si só, uma contribuição científica válida.

O artigo completo (formato SBC) está em [`paper/`](paper/) — veja também a
seção [Artigo](#artigo) abaixo.

---

## Sumário

- [Estrutura do repositório](#estrutura-do-repositório)
- [Dados](#dados)
- [Metodologia](#metodologia)
- [Principais resultados](#principais-resultados)
- [Figuras](#figuras)
- [Como reproduzir](#como-reproduzir)
- [Artigo](#artigo)
- [Limitações](#limitações)
- [Licença](#licença)

## Estrutura do repositório

```
.
├── dev/
│   ├── pipeline.py            # pipeline completo: dados -> features -> modelos -> figuras
│   ├── lstm_utils.py          # LSTM, splits walk-forward, treino/avaliação (reaproveitado)
│   ├── paper_style.py         # estilo visual compartilhado das figuras (paleta, fonte)
│   ├── regenerate_figures.py  # regenera as figuras 01-08 a partir de dados já cacheados
│   ├── make_extra_figures.py  # figura 09 (previsto vs. real, com RMSE anotado)
│   └── validate_synthetic.py  # smoke-test do pipeline com dados sintéticos (sem downloads)
├── data/
│   ├── raw/                   # cache local: OHLCV (yfinance) e notícias (Kaggle)
│   └── processed/             # cache: sentimento FinBERT, dataset de features final
├── figs/                      # figuras geradas (01 a 09)
├── outputs/                   # métricas e tabelas em CSV (baselines, testes estatísticos, etc.)
├── paper/                     # artigo em LaTeX (modelo SBC), pronto para Overleaf
└── README.md
```

## Dados

Duas fontes públicas e gratuitas:

- **Preço (OHLCV)** do BTC-USD via [`yfinance`](https://pypi.org/project/yfinance/)
  (API do Yahoo Finance), 2020 até a data de execução.
- **Notícias de criptomoedas**: dataset
  [`Cryptocurrency News`](https://www.kaggle.com/datasets/oliviervha/crypto-news)
  (Kaggle), ~31 mil manchetes com data de publicação, cobrindo out/2021–dez/2023.

O alinhamento temporal usa o sentimento agregado do dia `D` para prever o
fechamento de `D+1` (nunca o inverso), com checagem numérica explícita
anti-*look-ahead bias*. Como as notícias cobrem só ~33% do histórico de preço,
a modelagem é restrita à janela com cobertura real de notícias (799 dias).

## Metodologia

- **Sentimento:** FinBERT (`ProsusAI/finbert`), não VADER — mais adequado a
  linguagem financeira.
- **Features:** 14 técnicas (retornos, médias móveis, volatilidade, RSI de
  Wilder, variação de volume) + 8 de sentimento agregado por dia.
- **Split:** *walk-forward* de janela expandindo, 5 folds, treino < validação
  < teste sempre em ordem cronológica, sem sobreposição. `StandardScaler`
  ajustado só no treino de cada fold.
- **Baselines honestos:** persistência ingênua (*random walk*) e ARIMA
  univariado (ordem por AIC, atualizado dia a dia sem *refit*).
- **Modelo:** LSTM com encoder compartilhado e duas cabeças (regressão +
  classificação direcional), mesma arquitetura/hiperparâmetros para o modelo
  numérico puro e o numérico+sentimento — única variável isolada é o conjunto
  de features de entrada. 5 seeds por combinação (fold × features).
- **Teste estatístico:** teste $t$ pareado e Wilcoxon sobre RMSE, MAE,
  acurácia e F1 (25 pares fold×seed).
- **Interpretabilidade:** *permutation importance* sobre o último fold.
- **Trading:** simulação *long/short* com custo de 0,1% por operação,
  comparada a *buy-and-hold*.

## Principais resultados

| Modelo | RMSE (USD) | MAE (USD) | Acurácia | F1 |
|---|---|---|---|---|
| Persistência ingênua | 617,5 ± 134,3 | 410,0 ± 108,1 | 0,456 ± 0,023 | 0,441 ± 0,029 |
| ARIMA | 618,3 ± 134,3 | 406,8 ± 107,4 | 0,544 ± 0,023 | 0,543 ± 0,026 |
| LSTM numérico puro | 975,3 ± 309,6 | 758,5 ± 291,1 | 0,512 ± 0,050 | 0,293 ± 0,194 |
| LSTM numérico + sentimento | 1015,2 ± 202,0 | 783,8 ± 150,3 | 0,509 ± 0,040 | 0,382 ± 0,189 |

Simulação de trading (fold de teste final, custo de 0,1%/operação):
**numérico puro −36,4%**, **numérico+sentimento −40,2%**, **buy-and-hold
+64,3%** — ambos os LSTMs perderam para a estratégia passiva.

Tabelas completas em [`outputs/`](outputs/).

## Figuras

| # | Figura |
|---|---|
| 01 | Preço e volume diário do BTC-USD |
| 02 | Cobertura temporal das notícias vs. histórico de preço |
| 03 | Distribuição do sentimento (FinBERT) |
| 03b | Decomposição sazonal em 3 escalas (1, 30, 365 dias) |
| 04 | PCA exploratório (variância explicada) |
| 05 | Heatmap de correlação entre features-chave e o alvo |
| 06 | Esquema walk-forward (treino/validação/teste por fold) |
| 07 | Permutation importance |
| 08 | Simulação de trading (retorno acumulado) |
| 09 | Fechamento real vs. previsto, fora da amostra |

## Como reproduzir

```bash
pip install pandas numpy yfinance transformers torch scikit-learn \
            statsmodels matplotlib seaborn scipy
python dev/pipeline.py
```

O pipeline usa cache em disco (`data/raw/`, `data/processed/`) para preço,
notícias e sentimento FinBERT já processado — reexecuções subsequentes não
rebaixam nada da internet nem reprocessam o FinBERT, a menos que o cache seja
apagado manualmente. Para um smoke-test rápido da lógica, sem downloads
pesados, rode `python dev/validate_synthetic.py`.

## Artigo

O artigo completo, no modelo de artigo da SBC, está em
[`paper/main.tex`](paper/main.tex) (pronto para importar no
[Overleaf](https://www.overleaf.com/) — basta subir a pasta `paper/` como um
projeto novo).

## Limitações

Cobertura temporal restrita das notícias (~33% do histórico de preço); fonte
única de notícias (não inclui redes sociais); granularidade diária (não
intradiária); FinBERT não é específico para jargão cripto; amostra de 799
dias particionada em poucos folds; ativo único (BTC-USD). Discussão completa
na Seção 6 do artigo.

## Licença

MIT — veja [`LICENSE`](LICENSE).

---

Projeto desenvolvido como trabalho final da disciplina de IA.
Contato: `seu.email@instituicao.edu.br`
