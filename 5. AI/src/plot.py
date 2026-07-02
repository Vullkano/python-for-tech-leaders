
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D # Import necessário para 3D

from sklearn.svm import SVC
from matplotlib.lines import Line2D

from sklearn.neighbors import KNeighborsClassifier


# Centralized style to keep plots consistent across notebooks.
DEFAULT_STYLE = {
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "--",
}


def set_plot_style() -> None:
    """Apply a clean, high-contrast style for all plots."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    plt.rcParams.update(DEFAULT_STYLE)


def plot_bar(
    data: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
    palette: str = "deep" # Opções fixes: "viridis", "mako", "rocket", "deep"
) -> plt.Axes:
    """Barplot estilizado com contagem e percentagem no topo."""
    set_plot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Calcular totais para percentagens
    total = len(data)
    
    # TRUQUE para cores bonitas:
    # Se não houver 'hue' explícito, usamos o próprio 'x' como hue para colorir as barras
    # legend=False evita criar uma legenda desnecessária quando x=hue
    if hue:
        sns.countplot(data=data, x=x, hue=hue, palette=palette, ax=ax, edgecolor="white", linewidth=1.5)
    else:
        sns.countplot(data=data, x=x, hue=x, palette=palette, ax=ax, edgecolor="white", linewidth=1.5, legend=False)

    # Adicionar anotações inteligentes
    for p in ax.patches:
        height = p.get_height()
        # Se a barra for 0 ou NaN, ignora
        if not np.isfinite(height) or height == 0:
            continue
            
        percentage = f'{100 * height / total:.1f}%'
        
        # Cor do texto cinzento escuro para contraste profissional
        ax.annotate(f'{int(height)}\n({percentage})',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', # va='bottom' mete o texto EM CIMA da barra
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=11, weight='bold', color='#444444')

    # Limpeza visual (Estilo Clean)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([]) # Remover eixo Y (os números já estão nas barras)
    ax.set_ylabel("")
    ax.set_xlabel(x, fontsize=12, weight='bold', labelpad=10)
    
    if title: 
        ax.set_title(title, fontsize=16, weight="bold", loc='left', pad=20, color="#333333")
    
    return ax

def plot_hist(
    data: pd.Series | np.ndarray,
    bins: int = 25,
    kde: bool = True,
    color: str = "#3498db",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    show_mean_median: bool = True, # <--- NOVA FUNCIONALIDADE
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histograma com KDE e linhas automáticas de Média/Mediana."""
    set_plot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot principal
    sns.histplot(
        data, bins=bins, kde=False, color=color, ax=ax,
        alpha=0.7, edgecolor="white", linewidth=1.5, stat="density"
    )
    
    if kde:
        sns.kdeplot(data, color=color, ax=ax, linewidth=3, cut=0)

    # Adicionar linhas de Média e Mediana
    if show_mean_median:
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Média: {mean_val:.1f}')
        ax.axvline(median_val, color='#2ecc71', linestyle='-', linewidth=2, label=f'Mediana: {median_val:.1f}')
        ax.legend()

    # Estilização Clean
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False) # Remove eixo Y feio
    ax.yaxis.set_ticks([]) # Remove numeros eixo Y (densidade não é intuitiva para leigos)
    ax.grid(axis='y', alpha=0.2)
    
    if title: ax.set_title(title, fontsize=16, weight="bold", loc='left')
    if xlabel: ax.set_xlabel(xlabel, fontsize=12)
    
    return ax

def plot_confusion_matrix(
    y_true: Iterable,
    y_pred: Iterable,
    classes: list = ["Sobreviveu", "Não Sobreviveu"], # Atenção: A ordem tem de bater certo com 'labels'
    labels: list = [1, 0], # <--- O TEU NOVO PARÂMETRO AQUI
    title: str = "Matriz de Confusão",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (7, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plota a matriz de confusão.
    labels=[1, 0] -> TP no topo esquerdo, TN no fundo direito.
    labels=[0, 1] -> TN no topo esquerdo, TP no fundo direito.
    """
    
    # Gera a matriz usando a ordem definida em 'labels'
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum()
    
    # Define as etiquetas de texto com base na ordem dos labels
    if labels == [1, 0]:
        # Ordem Invertida (1 primeiro): TP, FN, FP, TN
        group_names = ['True Pos','False Neg','False Pos','True Neg']
    else:
        # Ordem Padrão (0 primeiro): TN, FP, FN, TP
        group_names = ['True Neg','False Pos','False Neg','True Pos']
    
    annot_labels = []
    iterator = 0
    
    for i in range(cm.shape[0]):
        row_labels = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            txt = f"{group_names[iterator]}\n{count}\n({pct:.1%})"
            iterator += 1
            row_labels.append(txt)
        annot_labels.append(row_labels)
    annot_labels = np.array(annot_labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")

    # Heatmap
    sns.heatmap(
        cm, annot=annot_labels, fmt='', cmap=cmap,
        linewidths=3, linecolor="white", cbar=False,
        square=True, ax=ax,
        annot_kws={"fontsize": 14, "weight": "bold"},
        vmin=0
    )

    # Estilização
    ax.set_title(title, fontsize=18, weight="bold", pad=20, color="#333333")
    ax.set_xlabel("Predito", fontsize=13, weight="bold", labelpad=10, color="#555555")
    ax.set_ylabel("Real", fontsize=13, weight="bold", labelpad=10, color="#555555")
    
    # Ajusta as labels dos eixos
    ax.set_xticklabels(classes, rotation=0, fontsize=12)
    ax.set_yticklabels(classes, rotation=0, fontsize=12, va="center")
    ax.tick_params(length=0) 

    return ax

def plot_corr_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "vlag",
    annot: bool = True,
    mask_upper: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Heatmap de correlação bonito e limpo."""
    set_plot_style()
    corr = df.corr()
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    else:
        mask = None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"shrink": .8},
        ax=ax,
        annot_kws={"fontsize": 12},
    )
    if title:
        ax.set_title(title)
    return ax

def plot_feature_importance(model, feature_names, title="Importância das Variáveis", color="#00A699"):
    """Mostra quais as variáveis que mais influenciaram o modelo."""
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16, weight='bold', pad=20)
    plt.barh(range(len(indices)), importances[indices], color=color, align='center', edgecolor='white', linewidth=1)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=12)
    plt.xlabel('Importância Relativa', fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

def plot_logistic_curve(model, X, y, title="Regressão Logística: A Curva Sigmoide"):
    """
    Plota a curva S perfeita usando a 'Decision Function' do modelo.
    Mostra os pontos reais no topo (1) e fundo (0) com as cores certas.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1. Calcular a "Pontuação" (Eixo X) e Probabilidade (Eixo Y)
    # decision_function dá-nos a soma matemática de todas as variáveis (w1*x1 + w2*x2 + ...)
    decision_values = model.decision_function(X)
    probs = model.predict_proba(X)[:, 1]

    # Ordenar valores para a linha ficar lisa e não riscada
    sort_idx = np.argsort(decision_values)
    x_line = decision_values[sort_idx]
    y_line = probs[sort_idx]

    plt.figure(figsize=(12, 7))

    # 2. Desenhar a Curva Sigmoide (O "S")
    plt.plot(x_line, y_line, color='#333333', linewidth=3, label='Probabilidade Calculada', zorder=2)

    # 3. Desenhar os Pontos Reais (Dados de Teste)
    # Adicionamos um pequeno ruído (jitter) vertical apenas para visualização
    # para os pontos não ficarem todos amontoados nas linhas 0 e 1
    y_jitter = y + np.random.normal(0, 0.02, size=len(y))
    
    # Cores manuais (Vermelho e Verde Água)
    colors = ['#FF5A5F' if val == 0 else '#00A699' for val in y]
    
    plt.scatter(decision_values, y_jitter, c=colors, alpha=0.6, s=50, edgecolor='white', linewidth=0.5, zorder=3)

    # 4. Decorações
    plt.axhline(0.5, color='#E74C3C', linestyle='--', alpha=0.7, label='Fronteira de Decisão (50%)')
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5) # Onde a pontuação é 0, a prob é 50%
    
    plt.title(title, fontsize=16, weight='bold', pad=15)
    plt.xlabel("Pontuação do Modelo (Combinação das Variáveis)", fontsize=12, labelpad=10)
    plt.ylabel("Probabilidade de Sobrevivência", fontsize=12, labelpad=10)
    plt.legend(loc='center right')
    
    # Textos explicativos no gráfico
    plt.text(x_line.min(), 0.05, "Zona de Morte\n(Alta certeza)", color='#FF5A5F', fontweight='bold')
    plt.text(x_line.max(), 0.90, "Zona de Vida\n(Alta certeza)", color='#00A699', fontweight='bold', ha='right')

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_svm_margins(model, X_train, y_train, title="SVM: Margens e Vetores de Suporte"):
    """
    Visualiza o SVM em 2D: Triângulos para Vetores de Suporte, Círculos para os restantes.
    """
    # 1. Preparar os dados
    X_mat = X_train.values if hasattr(X_train, 'values') else X_train
    y_mat = y_train.values if hasattr(y_train, 'values') else y_train

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_mat)
    
    # 2. Modelo de visualização 2D
    viz_model = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma)
    viz_model.fit(X_reduced, y_mat)
    
    # 3. Configurar cores e gráfico
    colors_hex = ['#FF5A5F', '#00A699']
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # --- SEPARAR VETORES DE SUPORTE DOS RESTANTES ---
    # Pegamos nos índices de quem é vetor de suporte
    support_indices = viz_model.support_
    mask_support = np.zeros(len(X_reduced), dtype=bool)
    mask_support[support_indices] = True

    # --- DESENHAR OS PONTOS NORMAIS (Círculos) ---
    for i, color in enumerate(colors_hex):
        # Filtra: classe i E NÃO é vetor de suporte
        idx = np.where((y_mat == i) & (~mask_support))
        ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], c=color, 
                   marker='o', s=40, edgecolors='k', alpha=0.5, label='_nolegend_')

    # --- DESENHAR OS VETORES DE SUPORTE (Triângulos) ---
    for i, color in enumerate(colors_hex):
        # Filtra: classe i E É vetor de suporte
        idx = np.where((y_mat == i) & (mask_support))
        ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], c=color, 
                   marker='^', s=80, edgecolors='black', linewidths=1.5, label='_nolegend_')

    # --- DESENHAR A ESTRADA (Fronteira e Margens) ---
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = viz_model.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, 
               linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    # 4. Legenda e Estética
    plt.title(title, fontsize=16, weight='bold', pad=15)
    plt.xlabel('Componente Principal 1 (PCA)', fontsize=12)
    plt.ylabel('Componente Principal 2 (PCA)', fontsize=12)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Não Sobreviveu (Normal)', markerfacecolor=colors_hex[0], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Sobreviveu (Normal)', markerfacecolor=colors_hex[1], markersize=8),
        Line2D([0], [0], marker='^', color='w', label='Vetor de Suporte (Crítico)', 
               markerfacecolor='gray', markeredgecolor='black', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, shadow=True)
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_knn_boundaries(model, X_train, y_train, title="KNN: Fronteiras de Decisão (Vizinhança)"):
    """
    Cria um mapa de cores mostrando as zonas onde o KNN classifica como Sobrevivente ou Não.
    """
    # 1. Preparar dados (PCA para 2D)
    X_mat = X_train.values if hasattr(X_train, 'values') else X_train
    y_mat = y_train.values if hasattr(y_train, 'values') else y_train
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_mat)
    
    # 2. Treinar modelo visual
    viz_model = KNeighborsClassifier(n_neighbors=model.n_neighbors)
    viz_model.fit(X_reduced, y_mat)
    
    # 3. Criar a malha (grid)
    h = .05  # Tamanho do passo na malha
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Prever para cada ponto da malha
    Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 4. Plotar
    plt.figure(figsize=(12, 7))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA']) # Cores suaves para o fundo
    cmap_bold = ['#FF5A5F', '#00A699'] # Cores fortes para os pontos

    # Desenhar as zonas
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.3)

    # Desenhar os pontos reais
    for i, color in enumerate(cmap_bold):
        idx = np.where(y_mat == i)
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], c=color, 
                    edgecolor='k', s=40, alpha=0.8)

    plt.title(f"{title} | K={model.n_neighbors}", fontsize=16, weight='bold')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.show()

def plot_relevancia_reg(modelo, colunas):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Extrair dados dependendo do tipo de modelo
    if hasattr(modelo, 'feature_importances_'):
        importancia = modelo.feature_importances_
        titulo = f"Relevância das Variáveis - {type(modelo).__name__}"
    elif hasattr(modelo, 'coef_'):
        importancia = np.abs(modelo.coef_)
        titulo = "Relevância das Variáveis (Pesos Absolutos) - Linear Regression"
    else:
        print("Modelo não suportado para este gráfico.")
        return

    # Criar DataFrame e ordenar
    df_importancia = pd.DataFrame({'Variável': colunas, 'Relevância': importancia})
    df_importancia = df_importancia.sort_values(by='Relevância', ascending=False)

    # Gerar Gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Relevância', y='Variável', data=df_importancia, palette='viridis')
    plt.title(titulo)
    plt.xlabel('Nível de Importância / Peso')
    plt.ylabel('Características do Imóvel')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()