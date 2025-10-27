import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, _tree
import plotly.graph_objects as go
import warnings
import io

warnings.filterwarnings('ignore')


# Кэширование для тяжелых функций
@st.cache_data
def comprehensive_clean_data_cached(df, remove_outliers=True, outlier_multiplier=1.5, numeric_columns=None):
    return comprehensive_clean_data(df, remove_outliers, outlier_multiplier, numeric_columns)


@st.cache_data
def encoded_features_cached(df, num_columns, cat_columns):
    return encoded_features(df, num_columns, cat_columns)


def show_info(df):
    """Функция для отображения основной информации о датафрейме"""
    if df is not None and not df.empty:
        st.markdown("---")
        st.subheader("📊 Основная информация о датафрейме")
        
        st.write("**1. Информация о структуре:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        st.write("**2. Статистическое описание числовых признаков:**")
        st.dataframe(df.describe())
        
        st.write("**3. Дополнительная информация:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Размер датафрейма", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            st.metric("Всего пропущенных значений", df.isnull().sum().sum())
        with col3:
            st.metric("Количество дубликатов", df.duplicated().sum())
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Числовые столбцы", numeric_cols)
        
        st.write("**4. Детальная информация о пропущенных значениях:**")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Колонка': missing_data.index,
            'Количество пропусков': missing_data.values,
            'Процент пропусков': missing_percent.values
        })
        missing_df = missing_df[missing_df['Количество пропусков'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.success("✅ Пропущенных значений нет")
    else:
        st.warning("Датасет не загружен или пустой")


def encoded_features(df, num_columns, cat_columns):
    """Кодирует числовые и категориальные признаки датафрейма"""
    df_encoded = df.copy()
    
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    all_features = num_columns + cat_columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), all_features)
        ])

    X_processed = preprocessor.fit_transform(df_encoded)

    return X_processed, label_encoders, preprocessor


def plot_dendogram_features(X_processed, feature_names):
    """Построение дендограммы признаков"""
    Z_features = linkage(X_processed.T, method='ward', metric='euclidean')

    fig, ax = plt.subplots(figsize=(16, 10))
    dendrogram(Z_features,
               labels=feature_names,
               leaf_rotation=90,
               leaf_font_size=10,
               show_contracted=True,
               color_threshold=0.7 * max(Z_features[:, 2]),
               ax=ax)

    ax.set_title('Дендрограмма кластеризации ПРИЗНАКОВ\n(группировка похожих фичей)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Реальные названия признаков', fontsize=12)
    ax.set_ylabel('Расстояние', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig


def calculate_inertia(Z, n_clusters, X_processed):
    """Строит инерцию для метода определения кол-ва кластеров с помощью локтя"""
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    inertia = 0
    for i in range(1, n_clusters + 1):
        cluster_points = X_processed[clusters == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia


def find_elbow_point(inertias, cluster_range):
    """Автоматически находит точку локтя на графике инерции"""
    first_diff = np.diff(inertias)
    second_diff = np.diff(first_diff)
    
    elbow_idx = np.argmax(np.abs(second_diff)) + 1  
    return list(cluster_range)[elbow_idx]


def comprehensive_clean_data(df, remove_outliers=True, outlier_multiplier=1.5, numeric_columns=None):
    """Комплексная очистка данных - всегда удаляем пропущенные значения"""
    df_clean = df.copy()
    
    initial_shape = df_clean.shape
    df_clean = df_clean.dropna()
    st.write(f"Удалено строк с пропусками: {initial_shape[0] - df_clean.shape[0]}")
    
    if remove_outliers and numeric_columns:
        initial_shape = df_clean.shape
        for col in numeric_columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_multiplier * IQR
                upper_bound = Q3 + outlier_multiplier * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        st.write(f"Удалено строк с выбросами: {initial_shape[0] - df_clean.shape[0]}")
    
    return df_clean


def delete_cat_with_zero_dispersion(df, cat_cols, threshold=0.95):
    """Удаляет категориальные столбцы с маленькой дисперсией значений"""
    cols_to_drop = []
    
    for col in cat_cols:
        if col in df.columns:
            value_counts = df[col].value_counts(normalize=True)
            most_common_ratio = value_counts.iloc[0] if len(value_counts) > 0 else 1.0
            
            if most_common_ratio >= threshold:
                cols_to_drop.append(col)
                st.write(f"Колонка '{col}' удалена: доля самого частого значения = {most_common_ratio:.3f}")
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.write(f"Удалено колонок: {len(cols_to_drop)}")
    else:
        st.write("Нет колонок с низкой дисперсией для удаления")
    
    return df


def plot_numeric_distributions(df, columns, cols=3):
    """Визуализация распределения числовых признаков"""
    n = len(columns)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            df[col].hist(ax=axes[i], bins=30, alpha=0.7)
            axes[i].set_title(f'Распределение {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Частота')
    
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_categorical_distributions(df, columns, cols=3):
    """Визуализация распределения категориальных признаков"""
    n = len(columns)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind='bar', ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Распределение {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Количество')
            axes[i].tick_params(axis='x', rotation=45)
    
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def get_node_samples(tree_clf, X, node_id):
    """Получает индексы образцов в узле дерева"""
    tree_ = tree_clf.tree_
    node_indicator = tree_clf.decision_path(X)
    node_array = node_indicator.toarray()
    return np.where(node_array[:, node_id] == 1)[0]


def plot_decision_tree_interactive(tree_clf, X, y, target_values, feature_names, class_names,
                                   base_width=1200, per_leaf_width=200, max_width=10000):
    """
    Интерактивная визуализация дерева решений с автонастройкой ширины,
    цветной разметкой ветвей (синяя — Да, красная — Нет)
    и корректной легендой по реально присутствующим кластерам.
    """

    tree_ = tree_clf.tree_

    # === Цвета кластеров ===
    cluster_colors = {
        'Cluster 1': '#FF6B6B',
        'Cluster 2': "#C44EF7",
        'Cluster 3': "#5CA968",
        'Cluster 4': "#DCD96E",
        'Cluster 5': "#F0BE52",
        'Cluster 6': '#DDA0DD',
        'Cluster 7': "#3B2E3C",
        'Cluster 8': "#755656",
        'Cluster 9': "#4C8598",
        'Cluster 10': "#3F543C"
    }

    # === Подсчёт числа листьев и глубины ===
    def compute_meta(node=0, depth=0):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            return 1, depth
        l_count, l_depth = compute_meta(tree_.children_left[node], depth + 1)
        r_count, r_depth = compute_meta(tree_.children_right[node], depth + 1)
        return l_count + r_count, max(l_depth, r_depth)

    total_leaves, max_depth = compute_meta(0, 0)

    # === Автоматическая настройка расстояния и ширины ===
    spacing_factor = 1.5 + np.log2(total_leaves + 1) * 0.3
    fig_width = int(min(max_width, base_width + per_leaf_width * total_leaves))

    # === In-order размещение узлов ===
    node_info = {}
    leaf_counter = {'i': 0}
    raw_gap = spacing_factor

    def assign_x_inorder(node, depth=0):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            i = leaf_counter['i']
            raw_x = i * raw_gap
            leaf_counter['i'] += 1
            node_info[node] = {'raw_x': raw_x, 'depth': depth, 'is_leaf': True}
            return raw_x
        else:
            lx = assign_x_inorder(tree_.children_left[node], depth + 1)
            rx = assign_x_inorder(tree_.children_right[node], depth + 1)
            mid = (lx + rx) / 2
            node_info[node] = {'raw_x': mid, 'depth': depth, 'is_leaf': False}
            return mid

    assign_x_inorder(0)

    xs = np.array([v['raw_x'] for v in node_info.values()])
    xmin, xmax = xs.min(), xs.max()
    scale = 1.0 / (xmax - xmin) if xmax > xmin else 1.0

    for nid, v in node_info.items():
        v['x'] = (v['raw_x'] - xmin) * scale
        v['y'] = 1.0 - v['depth'] / (max_depth + 1)

    # === получить индексы выборки для узла ===
    def get_node_samples(tree_clf, X, node_id):
        node_indicator = tree_clf.decision_path(X)
        return np.where(node_indicator[:, node_id].toarray().ravel() == 1)[0]

    # === текст и цвет узла ===
    def get_node_text_and_color(node_id):
        samples_count = tree_.n_node_samples[node_id]
        sample_idx = get_node_samples(tree_clf, X, node_id)
        
        target_mean = None
        if target_values is not None and len(sample_idx) > 0:
            target_mean = np.mean(target_values[sample_idx])
        
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            fname = feature_names[tree_.feature[node_id]]
            thr = tree_.threshold[node_id]
            
            # Обрезаем длинные названия признаков
            if len(fname) > 25:
                fname = fname[:22] + "..."
            
            # Безопасное получение столбца признака
            if isinstance(X, pd.DataFrame):
                col_values = X.iloc[:, tree_.feature[node_id]].values
            else:
                col_values = np.array(X)
                if col_values.ndim > 1:
                    col_values = col_values[:, tree_.feature[node_id]]
            
            # Проверка бинарного признака
            is_binary = np.all(np.isin(np.unique(col_values), [0, 1])) or len(np.unique(col_values)) == 2
            
            # Логика текста узла
            if is_binary and abs(thr - 0.5) < 1e-6:
                node_text = f"{fname}"
            else:
                node_text = f"{fname} ≤ {thr:.2f}"
            
            lines = [node_text, f"Samples: {samples_count}"]
            if target_mean is not None:
                lines.append(f"Mean: {target_mean:.2f}")
            
            # Цвета для непрерывных/регрессионных узлов
            color, border = "rgba(216,216,218,0.8)", "rgba(120,120,120,0.8)"
        
        else:
            cls_idx = int(np.argmax(tree_.value[node_id]))
            cname = class_names[cls_idx]
            lines = [f"{cname}", f"Samples: {samples_count}"]
            if target_mean is not None:
                lines.append(f"Mean: {target_mean:.2f}")
            
            # Цвета для классификационных узлов
            color = cluster_colors.get(cname, 'lightgreen')
            border = 'darkgreen'
        
        return lines, color, border

    # === Список реально присутствующих кластеров ===
    present_classes = []
    for nid, info in node_info.items():
        if info['is_leaf']:
            cname = class_names[int(np.argmax(tree_.value[nid]))]
            if cname not in present_classes:
                present_classes.append(cname)

    # === Построение графика ===
    fig = go.Figure()

    rect_w, rect_h = 0.08, 0.05

    def add_nodes_and_edges(node):
        info = node_info[node]
        x, y = info['x'], info['y']
        text, color, border = get_node_text_and_color(node)
        rect_h_local = rect_h + (len(text) - 1) * 0.012

        fig.add_shape(
            type="rect",
            x0=x - rect_w / 2, y0=y - rect_h_local / 2,
            x1=x + rect_w / 2, y1=y + rect_h_local / 2,
            line=dict(color=border, width=2),
            fillcolor=color, opacity=0.95
        )

        for i, t in enumerate(text):
            ty = y + rect_h_local / 2 - (i + 1) * rect_h_local / (len(text) + 1)
            fig.add_annotation(x=x, y=ty, text=t, showarrow=False,
                               font=dict(size=11, color='black'), align='center')

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            left, right = tree_.children_left[node], tree_.children_right[node]
            lx, ly = node_info[left]['x'], node_info[left]['y']
            rx, ry = node_info[right]['x'], node_info[right]['y']

            # ✅ Синие линии для "Да" (левая ветвь), красные для "Нет" (правая ветвь)
            fig.add_trace(go.Scatter(
                x=[x, lx], y=[y - rect_h_local / 2, ly + rect_h / 2],
                mode='lines', line=dict(width=2, color='blue'),
                hoverinfo='none', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x, rx], y=[y - rect_h_local / 2, ry + rect_h / 2],
                mode='lines', line=dict(width=2, color='red'),
                hoverinfo='none', showlegend=False
            ))

            # Подписи "Да"/"Нет"
            midx_left, midy_left = (x + lx) / 2, (y + ly) / 2
            midx_right, midy_right = (x + rx) / 2, (y + ry) / 2
            fig.add_annotation(x=midx_left, y=midy_left, text="Да", showarrow=False,
                               font=dict(size=10, color='blue'))
            fig.add_annotation(x=midx_right, y=midy_right, text="Нет", showarrow=False,
                               font=dict(size=10, color='red'))

            add_nodes_and_edges(left)
            add_nodes_and_edges(right)

    add_nodes_and_edges(0)

    # === Легенда по реально используемым кластерам ===
    for cname in present_classes:
        color = cluster_colors.get(cname, '#CCCCCC')
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=14, color=color, line=dict(width=1, color='black')),
            name=cname
        ))

    # === Добавим легенду для ветвей "Да"/"Нет" ===
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(width=2, color='blue'),
        name='Да (≤ порога)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(width=2, color='red'),
        name='Нет (> порога)'
    ))

    # === Оформление ===
    fig.update_layout(
        title=dict(
            text='<b>Дерево решений для предсказания кластеров</b>',
            x=0.4, font=dict(size=20)
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.05, 1.15]),
        width=fig_width,
        height=1300,
        plot_bgcolor='white',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title='Обозначения',
            x=1.02, y=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(color='black', size=12)
        )
    )

    return fig


def plot_decision_tree_with_target(
        tree_clf, X, y, target_values,
        feature_names, class_names):
    """Основная функция визуализации"""
    return plot_decision_tree_interactive(
        tree_clf, X, y, target_values,
        feature_names, class_names)


# Основное приложение Streamlit
def main():
    st.set_page_config(page_title="Clustering Analysis App", layout="wide")
    st.title("📊 Анализ данных и кластеризация")
    
    # Инициализация session_state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'clustered_df' not in st.session_state:
        st.session_state.clustered_df = None
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'tree_built' not in st.session_state:
        st.session_state.tree_built = False
    if 'optimal_clusters' not in st.session_state:
        st.session_state.optimal_clusters = None
    if 'X_processed' not in st.session_state:
        st.session_state.X_processed = None
    if 'num_cols' not in st.session_state:
        st.session_state.num_cols = None
    if 'cat_cols' not in st.session_state:
        st.session_state.cat_cols = None
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    # Инициализация для графиков
    if 'preprocessing_figures' not in st.session_state:
        st.session_state.preprocessing_figures = {
            'numeric_distributions': None,
            'categorical_distributions': None
        }
    if 'preprocessing_completed' not in st.session_state:
        st.session_state.preprocessing_completed = False
    if 'show_preprocessing_results' not in st.session_state:
        st.session_state.show_preprocessing_results = False
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = {
            'dendrogram_fig': None,
            'cluster_dist_fig': None,
            'pca_fig': None,
            'numeric_summary': None,
            'categorical_summary': None,
            'target_summary': None
        }
    if 'show_clustering_results' not in st.session_state:
        st.session_state.show_clustering_results = False
    if 'tree_results' not in st.session_state:
        st.session_state.tree_results = {
            'tree_fig': None,
            'importance_fig': None,
            'feature_importance': None,
            'tree_data_info': None
        }
    if 'show_tree_results' not in st.session_state:
        st.session_state.show_tree_results = False

    # Загрузка данных
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV или Excel файл", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ Данные успешно загружены! Размер: {df.shape}")
            
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head())

            show_info(df)
            
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Выбор столбцов
        st.header("2. Выбор столбцов для анализа")
        
        all_columns = df.columns.tolist()
        
        col1, col2 = st.columns(2)

        with col2:
            st.subheader("Целевая переменная")
            target_options = [None] + all_columns
            selected_target = st.selectbox(
                "Выберите целевую переменную (опционально):",
                options=target_options,
                index=0,
                key="target_selector"
            )

            if selected_target:
                target_dtype = df[selected_target].dtype
                is_numeric = pd.api.types.is_numeric_dtype(df[selected_target])
                
                if is_numeric:
                    st.success(f"✅ Целевая переменная: **{selected_target}**")
                    st.info(f"Тип данных: {target_dtype}")
                    
                    target_stats = df[selected_target].describe()
                    st.write("**Базовая статистика:**")
                    st.write(f"- Минимум: {target_stats['min']:.2f}")
                    st.write(f"- Максимум: {target_stats['max']:.2f}")
                    st.write(f"- Среднее: {target_stats['mean']:.2f}")
                    st.write(f"- Медиана: {target_stats['50%']:.2f}")
                else:
                    st.error(f"❌ Целевая переменная '{selected_target}' должна быть числовой!")

        with col1:
            st.subheader("Входные признаки")
            
            available_features = all_columns.copy()
            if selected_target and selected_target in available_features:
                available_features.remove(selected_target)
            
            if len(available_features) > 0:
                default_features = available_features[:min(5, len(available_features))]
            else:
                default_features = []
            
            selected_features = st.multiselect(
                "Выберите признаки для анализа:",
                options=available_features,
                default=default_features,
                help="Таргетная переменная автоматически исключена из списка",
                key="features_selector"
            )
            
            if len(selected_features) == 0:
                st.error("❌ Необходимо выбрать хотя бы один признак!")
            elif len(selected_features) == 1:
                st.warning("⚠️ Рекомендуется выбрать несколько признаков для кластеризации")
            else:
                st.success(f"✅ Выбрано признаков: {len(selected_features)}")
        
        # Предобработка данных
        st.header("3. Предобработка данных")
        
        preprocessing_col1, preprocessing_col2 = st.columns(2)
        
        with preprocessing_col1:
            st.info("ℹ️ Пропущенные значения всегда удаляются автоматически")
            remove_outliers = st.checkbox("Удалить выбросы", value=True, key="outliers_checkbox")
        
        with preprocessing_col2:
            low_disp_threshold = st.slider(
                "Порог для удаления низкодисперсных категориальных признаков:",
                min_value=0.8, max_value=1.0, value=0.95, step=0.01,
                key="dispersion_slider"
            )
            outlier_multiplier = st.slider(
                "Множитель для определения выбросов (IQR):",
                min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                key="outlier_slider"
            )
        
        if st.button("Запустить предобработку", key="preprocessing_button"):
            if len(selected_features) == 0:
                st.error("❌ Нельзя запустить предобработку без выбранных признаков!")
            else:
                with st.spinner("Выполняется предобработка данных..."):
                    if selected_target and selected_target in selected_features:
                        features_to_use = [f for f in selected_features if f != selected_target]
                        st.warning(f"⚠️ Таргетная переменная '{selected_target}' исключена из входных признаков")
                    else:
                        features_to_use = selected_features.copy()
                    
                    if selected_target:
                        processed_df = df[[selected_target] + features_to_use].copy()
                    else:
                        processed_df = df[features_to_use].copy()
                    
                    num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if selected_target and selected_target in num_cols:
                        num_cols.remove(selected_target)
                    if selected_target and selected_target in cat_cols:
                        cat_cols.remove(selected_target)
                    
                    numeric_for_cleaning = num_cols.copy()
                    if selected_target and processed_df[selected_target].dtype in [np.number]:
                        numeric_for_cleaning.append(selected_target)
                    
                    processed_df = comprehensive_clean_data_cached(
                        processed_df,
                        remove_outliers=remove_outliers,
                        outlier_multiplier=outlier_multiplier,
                        numeric_columns=numeric_for_cleaning
                    )
                    
                    if cat_cols:
                        processed_df = delete_cat_with_zero_dispersion(
                            processed_df, cat_cols, threshold=low_disp_threshold
                        )
                        cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
                        if selected_target and selected_target in cat_cols:
                            cat_cols.remove(selected_target)
                    
                    st.session_state.processed_df = processed_df
                    st.session_state.num_cols = num_cols
                    st.session_state.cat_cols = cat_cols
                    st.session_state.selected_target = selected_target
                    st.session_state.selected_features = selected_features
                    st.session_state.clustering_done = False
                    st.session_state.tree_built = False
                    st.session_state.preprocessing_completed = True
                    st.session_state.show_preprocessing_results = True
                    st.session_state.show_clustering_results = False
                    
                    st.success(f"✅ Предобработка завершена! Новый размер: {processed_df.shape}")
                    
                    if num_cols:
                        fig_num = plot_numeric_distributions(processed_df, num_cols)
                        st.session_state.preprocessing_figures['numeric_distributions'] = fig_num
                    
                    if cat_cols:
                        fig_cat = plot_categorical_distributions(processed_df, cat_cols)
                        st.session_state.preprocessing_figures['categorical_distributions'] = fig_cat
        
        # Показываем результаты предобработки
        if st.session_state.preprocessing_completed and st.session_state.show_preprocessing_results:
            processed_df = st.session_state.processed_df
            num_cols = st.session_state.num_cols
            cat_cols = st.session_state.cat_cols
            
            st.subheader("Статистика после очистки")
            st.dataframe(processed_df.describe())
            
            if st.session_state.preprocessing_figures['numeric_distributions'] and num_cols:
                st.subheader("Распределения числовых признаков")
                st.pyplot(st.session_state.preprocessing_figures['numeric_distributions'])
            
            if st.session_state.preprocessing_figures['categorical_distributions'] and cat_cols:
                st.subheader("Распределения категориальных признаков")
                st.pyplot(st.session_state.preprocessing_figures['categorical_distributions'])
        
        # Кластеризация
        if st.session_state.processed_df is not None:
            st.header("4. Кластеризация")
            
            processed_df = st.session_state.processed_df
            num_cols = st.session_state.num_cols
            cat_cols = st.session_state.cat_cols
            
            cluster_col1, cluster_col2 = st.columns(2)
            
            with cluster_col1:
                st.subheader("Настройки кластеризации")
                use_auto_clusters = st.checkbox("Автоматический выбор числа кластеров (elbow method)", 
                                              value=True, key="auto_clusters_checkbox")
                
                if not use_auto_clusters:
                    n_clusters = st.slider("Число кластеров:", min_value=2, max_value=10, value=4,
                                         key="n_clusters_slider")
                else:
                    n_clusters = None
            
            with cluster_col2:
                st.subheader("Информация о данных")
                st.write(f"Числовые признаки: {len(num_cols)}")
                st.write(f"Категориальные признаки: {len(cat_cols)}")
                st.write(f"Общее количество образцов: {len(processed_df)}")
                if st.session_state.selected_target:
                    st.write(f"Целевая переменная: **{st.session_state.selected_target}**")
            
            if st.button("Запустить кластеризацию", key="clustering_button"):
                with st.spinner("Выполняется кластеризация..."):
                    try:
                        features_for_clustering = num_cols + cat_cols
                        X_processed, label_encoders, preprocessor = encoded_features_cached(
                            processed_df[features_for_clustering], num_cols, cat_cols
                        )
                        
                        st.subheader("Дендрограмма признаков")
                        dendro_fig = plot_dendogram_features(X_processed, features_for_clustering)
                        st.session_state.clustering_results['dendrogram_fig'] = dendro_fig
                        st.pyplot(dendro_fig)
                        
                        Z = linkage(X_processed, method='ward', metric='euclidean')
                        
                        if use_auto_clusters:
                            inertias = []
                            cluster_range = range(2, min(11, len(processed_df)))
                            
                            for k in cluster_range:
                                inertia = calculate_inertia(Z, k, X_processed)
                                inertias.append(inertia)
                            
                            optimal_clusters = find_elbow_point(inertias, cluster_range)
                            st.info(f"ℹ️ Автоматически найдено оптимальное число кластеров: {optimal_clusters}")
                        else:
                            optimal_clusters = n_clusters
                        
                        clusters = fcluster(Z, optimal_clusters, criterion='maxclust')
                        clustered_df = processed_df.copy()
                        clustered_df['Cluster'] = clusters
                        
                        st.session_state.clustered_df = clustered_df
                        st.session_state.optimal_clusters = optimal_clusters
                        st.session_state.X_processed = X_processed
                        st.session_state.clustering_done = True
                        st.session_state.tree_built = False  
                        st.session_state.show_tree_results = False
                        st.session_state.show_clustering_results = True
                        
                        st.success("✅ Кластеризация завершена!")
                        
                    except Exception as e:
                        st.error(f"Ошибка при кластеризации: {e}")
            
            # ВСЕГДА показываем результаты кластеризации если они есть
            if st.session_state.clustering_done:
                clustered_df = st.session_state.clustered_df
                optimal_clusters = st.session_state.optimal_clusters
                
                st.subheader("Результаты кластеризации")
                
                # Распределение по кластерам
                cluster_dist = clustered_df['Cluster'].value_counts().sort_index()
                fig_cluster_dist, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                         '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
                
                bars = ax.bar(cluster_dist.index, cluster_dist.values, 
                            color=colors[:optimal_clusters], 
                            edgecolor='darkred', alpha=0.7)
                
                for bar in bars:
                    height = bar.get_height()
                    percentage = (height / len(clustered_df)) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}\n({percentage:.1f}%)', 
                           ha='center', va='bottom', fontsize=9)
                
                ax.set_xlabel('Кластер')
                ax.set_ylabel('Количество образцов')
                ax.set_title(f'Распределение по кластерам ({optimal_clusters} кластеров)')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig_cluster_dist)
                
                # PCA визуализация
                if st.session_state.X_processed is not None:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(st.session_state.X_processed)
                    
                    fig_pca, ax = plt.subplots(figsize=(12, 8))
                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                       c=clustered_df['Cluster'], 
                                       cmap='tab10', alpha=0.7, s=60,
                                       edgecolors='w', linewidth=0.5)
                    
                    plt.colorbar(scatter, ax=ax, label='Кластер')
                    ax.set_title(f'Визуализация кластеров с помощью PCA\n({optimal_clusters} кластеров)')
                    ax.set_xlabel(f'Главная компонента 1 ({pca.explained_variance_ratio_[0]:.2%})')
                    ax.set_ylabel(f'Главная компонента 2 ({pca.explained_variance_ratio_[1]:.2%})')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig_pca)
                
                # Статистика по числовым признакам
                if st.session_state.num_cols:
                    numeric_summary = clustered_df.groupby('Cluster')[st.session_state.num_cols].mean()
                    st.write("**Статистика числовых признаков по кластерам:**")
                    st.dataframe(numeric_summary.style.background_gradient(cmap='Blues'))
                
                # Статистика по целевой переменной
                if st.session_state.selected_target:
                    target_summary = clustered_df.groupby('Cluster')[st.session_state.selected_target].agg([
                        'count', 'mean', 'std', 'min', 'max'
                    ]).round(2)
                    target_summary['count_pct'] = (target_summary['count'] / len(clustered_df) * 100).round(1)
                    
                    st.subheader(f"📊 Статистика целевой переменной '{st.session_state.selected_target}' по кластерам")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего кластеров", optimal_clusters)
                    with col2:
                        st.metric("Общее количество образцов", len(clustered_df))
                    with col3:
                        overall_mean = clustered_df[st.session_state.selected_target].mean()
                        st.metric("Общее среднее значение", f"{overall_mean:.2f}")
                    
                    st.dataframe(target_summary.style.background_gradient(cmap='YlOrBr', subset=['mean']))
                    
                    fig_target, ax = plt.subplots(figsize=(10, 6))
                    clusters_sorted = target_summary.sort_values('mean').index
                    means_sorted = target_summary.loc[clusters_sorted, 'mean']
                    
                    bars = ax.bar(range(len(means_sorted)), means_sorted, 
                                color=colors[:optimal_clusters], 
                                edgecolor='navy', alpha=0.7)
                    
                    overall_mean = clustered_df[st.session_state.selected_target].mean()
                    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
                              label=f'Общее среднее: {overall_mean:.2f}')
                    
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
                    
                    ax.set_xlabel('Кластер')
                    ax.set_ylabel(f'Среднее значение {st.session_state.selected_target}')
                    ax.set_title(f'Средние значения целевой переменной по кластерам\n({st.session_state.selected_target})')
                    ax.set_xticks(range(len(means_sorted)))
                    ax.set_xticklabels([f'Кластер {i}' for i in clusters_sorted])
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_target)
                
                # Статистика по категориальным признакам
                if st.session_state.cat_cols:
                    st.write("**Распределение категориальных признаков по кластерам:**")
                    for cat_col in st.session_state.cat_cols:
                        cross_tab = pd.crosstab(clustered_df['Cluster'], 
                                               clustered_df[cat_col], 
                                               normalize='index') * 100
                        st.write(f"**{cat_col}:**")
                        st.dataframe(cross_tab.style.background_gradient(cmap='Blues', axis=0))
                
                # Дерево решений
                st.header("5. Интерпретация кластеров с помощью дерева решений")
                
                tree_col1, tree_col2 = st.columns(2)
                
                with tree_col1:
                    tree_max_depth = st.slider("Максимальная глубина дерева:", 
                                             min_value=2, max_value=6, value=4,
                                             key="tree_depth_slider")
                    min_samples_leaf = st.slider("Минимальное количество образцов в листе:", 
                                               min_value=1, max_value=50, value=10,
                                               key="min_samples_leaf")
                
                with tree_col2:
                    show_target_info = st.checkbox("Показать среднее значение целевой переменной в узлах",
                                                 value=True,
                                                 help="Добавляет информацию о среднем значении целевой переменной в подписях узлов дерева")
                    st.info(f"Всего данных: {len(clustered_df)} образцов")
                    st.success("✅ Дерево будет использовать ВСЕ данные")
                
                if st.button("Построить дерево решений", key="tree_button"):
                    with st.spinner("Строим дерево решений..."):
                        try:
                            X_for_tree = clustered_df[st.session_state.num_cols + st.session_state.cat_cols].copy()
                            y_clusters = clustered_df['Cluster']
                            
                            X_encoded = pd.get_dummies(X_for_tree, columns=st.session_state.cat_cols, drop_first=True)
                            
                            X_train = X_encoded
                            y_train = y_clusters
                            
                            st.info(f"ℹ️ Используются ВСЕ данные: {len(X_train)} образцов")
                            
                            tree_clf = DecisionTreeClassifier(
                                max_depth=tree_max_depth,
                                min_samples_split=20,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42
                            )
                            
                            tree_clf.fit(X_train, y_train)
                            
                            target_values = None
                            if show_target_info and st.session_state.selected_target:
                                target_values = clustered_df[st.session_state.selected_target].values
                            
                            st.subheader("Визуализация дерева решений")
                            
                            # Используем интерактивную визуализацию
                            fig_tree = plot_decision_tree_with_target(
                                tree_clf,
                                X_train,
                                y_train,
                                target_values,
                                feature_names=X_encoded.columns.tolist(),
                                class_names=[f'Cluster {i}' for i in range(1, optimal_clusters+1)]
                            )
                            
                            st.session_state.tree_results['tree_fig'] = fig_tree
                            st.plotly_chart(fig_tree, use_container_width=True)
                            
                            st.subheader("Важность признаков")
                            feature_importance = pd.DataFrame({
                                'feature': X_encoded.columns,
                                'importance': tree_clf.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            st.write("Топ-10 наиболее важных признаков:")
                            feature_importance_reset = feature_importance.head(10).reset_index(drop=True)
                            st.session_state.tree_results['feature_importance'] = feature_importance_reset
                            st.dataframe(feature_importance_reset)
                            
                            fig_importance, ax = plt.subplots(figsize=(10, 6))
                            top_features = feature_importance.head(10)
                            top_features_sorted = top_features.sort_values('importance', ascending=True)
                            ax.barh(top_features_sorted['feature'], top_features_sorted['importance'], 
                                   color='skyblue', edgecolor='navy')
                            ax.set_xlabel('Важность признака')
                            ax.set_title('Топ-10 наиболее важных признаков для определения кластеров')
                            ax.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            st.session_state.tree_results['importance_fig'] = fig_importance
                            st.pyplot(fig_importance)
                            
                            st.session_state.tree_results['tree_data_info'] = {
                                'total_size': len(clustered_df),
                                'show_target_info': show_target_info
                            }
                            
                            st.session_state.tree_built = True
                            st.session_state.show_tree_results = True
                            
                        except Exception as e:
                            st.error(f"Ошибка при построении дерева: {e}")
            
            # Скачивание результатов
            if st.session_state.clustering_done:
                st.header("6. Скачивание результатов")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    st.session_state.clustered_df.to_excel(writer, sheet_name='Clustered_Data', index=False)
                    
                    if st.session_state.num_cols or st.session_state.cat_cols:
                        cluster_stats = st.session_state.clustered_df.groupby('Cluster').agg({
                            **{col: 'mean' for col in st.session_state.num_cols},
                            **{col: lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A' for col in st.session_state.cat_cols}
                        })
                        cluster_stats.to_excel(writer, sheet_name='Cluster_Statistics')
                    
                    if st.session_state.selected_target:
                        target_summary = st.session_state.clustered_df.groupby('Cluster')[st.session_state.selected_target].agg([
                            'count', 'mean', 'std', 'min', 'max'
                        ]).round(2)
                        target_summary['count_pct'] = (target_summary['count'] / len(st.session_state.clustered_df) * 100).round(1)
                        target_summary.to_excel(writer, sheet_name='Target_Statistics')
                
                output.seek(0)
                
                st.download_button(
                    label="📥 Скачать результаты кластеризации (Excel)",
                    data=output,
                    file_name=f"clustering_results_{st.session_state.optimal_clusters}_clusters.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_button"
                )


if __name__ == "__main__":
    main()
