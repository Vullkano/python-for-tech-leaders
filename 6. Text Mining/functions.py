from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
import re

RANDOM_STATE = 42

def show_wordcloud(text, title):
    text = str(text).strip()
    if not text:
        print(f"No text available for: {title}")
        return

    wc = WordCloud(
        width=1200,
        height=550,
        background_color="white",
        collocations=False,
        random_state=RANDOM_STATE
    ).generate(text)

    plt.figure(figsize=(14, 6))
    plt.imshow(np.array(wc.to_image()), interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

def compute_tf_df(doc_series):
    """
    TF = total number of times a word appears in the corpus
    DF = number of documents in which the word appears
    """
    docs = doc_series.fillna("").astype(str).tolist()

    tf = Counter()
    dfreq = Counter()

    for doc in docs:
        tokens = doc.split()
        tf.update(tokens)           # term frequency
        dfreq.update(set(tokens))   # document frequency

    return tf, dfreq, len(docs)

def make_df_color_func(dfreq):
    """
    Returns a color function for WordCloud where:
    - darker blue = lower document frequency
    - brighter yellow/green = higher document frequency
    """
    cmap = plt.cm.viridis

    min_df = min(dfreq.values()) if dfreq else 1
    max_df = max(dfreq.values()) if dfreq else 1
    denom = max(max_df - min_df, 1)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        df_value = dfreq.get(word, min_df)
        norm_value = (df_value - min_df) / denom
        r, g, b, _ = cmap(norm_value)
        return (int(r * 255), int(g * 255), int(b * 255))

    return color_func

def show_wordcloud_tf_size_df_color(doc_series, title):
    tf, dfreq, n_docs = compute_tf_df(doc_series)

    if not tf:
        print(f"No text available for: {title}")
        return

    wc = WordCloud(
        width=1400,
        height=650,
        background_color="white",
        collocations=False,
        random_state=RANDOM_STATE
    ).generate_from_frequencies(tf)   # size based on term frequency

    wc.recolor(color_func=make_df_color_func(dfreq))  # color based on document frequency

    plt.figure(figsize=(16, 7))
    plt.imshow(np.array(wc.to_image()), interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

    print(f"{title}")
    print(f"- Number of documents: {n_docs}")
    print(f"- Vocabulary size: {len(tf)}")
    print(f"- Word size encodes: term frequency (TF)")
    print(f"- Word color encodes: document frequency (DF)")


from collections import Counter

def plot_feature_selection_scatter(
    df,
    text_column,
    vectorizer,
    stopwords,
    max_points=8000,
    jitter=1.2,
    min_word_len=2,
    max_word_len=30,
    examples_stopwords=8,
    examples_selected=10,
    examples_not_selected=10,
    random_state=42,
    title="Vocabulário do corpus: stopwords vs features selecionadas"
):
    """
    Cria uma nuvem de pontos interativa para visualizar:
    - stopwords removidas
    - palavras dentro das features selecionadas pelo CountVectorizer
    - palavras fora das features selecionadas

    Parâmetros principais:
    df : pandas.DataFrame
        Dataset com a coluna de texto.
    text_column : str
        Nome da coluna que contém o texto bruto.
    vectorizer : CountVectorizer já treinado
        Exemplo: count_vectorizer depois de fit_transform().
    stopwords : set
        Conjunto de stopwords usadas no preprocessamento.
    max_points : int
        Número máximo de palavras a mostrar no gráfico.
    jitter : float
        Intensidade do espalhamento no eixo Y.
    """

    # Features escolhidas pelo CountVectorizer
    selected_features = set(vectorizer.get_feature_names_out())

    # Limpeza semelhante ao preprocessamento, mas sem remover stopwords
    def clean_without_removing_stopwords(text):
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    clean_text = df[text_column].fillna("").astype(str).apply(clean_without_removing_stopwords)

    # Frequência das palavras no corpus
    all_tokens = " ".join(clean_text).split()
    word_freq = Counter(all_tokens)

    rows = []

    for word, freq in word_freq.items():
        if word in stopwords or len(word) <= min_word_len:
            category = "Stopword removida"
        elif word in selected_features:
            category = "Dentro das features selecionadas"
        else:
            category = "Fora das features selecionadas"

        rows.append({
            "word": word,
            "frequency": freq,
            "category": category
        })

    scatter_df = pd.DataFrame(rows)

    # Filtrar palavras demasiado curtas/longas
    scatter_df = scatter_df[
        scatter_df["word"].str.len().between(min_word_len, max_word_len)
    ].copy()

    # Ordenar por frequência
    scatter_df = scatter_df.sort_values("frequency", ascending=False).reset_index(drop=True)

    # Jitter no eixo Y
    np.random.seed(random_state)
    scatter_df["y_jitter"] = np.random.uniform(-jitter, jitter, size=len(scatter_df))

    # Limitar pontos para não pesar demasiado
    scatter_plot_df = scatter_df.head(max_points).copy()

    # Cores
    color_map = {
        "Stopword removida": "red",
        "Dentro das features selecionadas": "green",
        "Fora das features selecionadas": "blue"
    }

    # Escolher exemplos espalhados por frequência
    def choose_label_examples(data, category, n_examples):
        subset = data[data["category"] == category].copy()

        if subset.empty:
            return subset

        subset = subset.sort_values("frequency", ascending=False)
        positions = np.linspace(
            0,
            len(subset) - 1,
            min(n_examples, len(subset))
        ).astype(int)

        return subset.iloc[positions].copy()

    label_examples = pd.concat([
        choose_label_examples(scatter_plot_df, "Stopword removida", examples_stopwords),
        choose_label_examples(scatter_plot_df, "Dentro das features selecionadas", examples_selected),
        choose_label_examples(scatter_plot_df, "Fora das features selecionadas", examples_not_selected)
    ])

    # Labels só para alguns exemplos
    scatter_plot_df["label"] = ""
    scatter_plot_df.loc[
        scatter_plot_df["word"].isin(label_examples["word"]),
        "label"
    ] = scatter_plot_df["word"]

    # Gráfico Plotly
    fig = px.scatter(
        scatter_plot_df,
        x="frequency",
        y="y_jitter",
        color="category",
        color_discrete_map=color_map,
        hover_name="word",
        hover_data={
            "word": True,
            "frequency": True,
            "category": True,
            "y_jitter": False,
            "label": False
        },
        text="label",
        log_x=True,
        title=title
    )

    fig.update_traces(
        marker=dict(size=7, opacity=0.55),
        textposition="top center"
    )

    fig.update_layout(
        width=1100,
        height=650,
        xaxis_title="Frequência da palavra no corpus, escala log",
        yaxis_title="Jitter artificial, apenas para espalhar os pontos",
        legend_title_text="Categoria",
        hovermode="closest"
    )

    fig.show()

    return scatter_df, label_examples