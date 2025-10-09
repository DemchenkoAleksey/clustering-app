import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import io

# Добавляем кэширование для тяжелых функций
@st.cache_data
def comprehensive_clean_data_cached(df, remove_outliers=True, outlier_multiplier=1.5, numeric_columns=None):
    """Кэшированная версия функции очистки данных"""
    return comprehensive_clean_data(df, remove_outliers, outlier_multiplier, numeric_columns)

@st.cache_data
def encoded_features_cached(df, num_columns, cat_columns):
    """Кэшированная версия функции кодирования"""
    return encoded_features(df, num_columns, cat_columns)

def show_info(df):
    """
    Функция для отображения основной информации о датафрейме в Streamlit
    """
    if df is not None and not df.empty:
        st.markdown("---")
        st.subheader("📊 Основная информация о датафрейме")
        
        # Информация о структуре
        st.write("**1. Информация о структуре:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        # Статистическое описание
        st.write("**2. Статистическое описание числовых признаков:**")
        st.dataframe(df.describe())
        
        # Дополнительная информация
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
        
        # Детальная информация о пропущенных значениях
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
    
    # Применяем LabelEncoder к каждой категориальной колонке
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Теперь все признаки числовые
    all_features = num_columns + cat_columns

    # Масштабируем все признаки
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
    
    # Находим точку максимального изгиба
    elbow_idx = np.argmax(np.abs(second_diff)) + 1  
    
    return list(cluster_range)[elbow_idx]

def comprehensive_clean_data(df, remove_outliers=True, 
                           outlier_multiplier=1.5, numeric_columns=None):
    """Комплексная очистка данных - всегда удаляем пропущенные значения"""
    df_clean = df.copy()
    
    # Всегда удаляем пропущенные значения
    initial_shape = df_clean.shape
    df_clean = df_clean.dropna()
    st.write(f"Удалено строк с пропусками: {initial_shape[0] - df_clean.shape[0]}")
    
    # Удаление выбросов (опционально)
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
    
    # Скрываем пустые subplots
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
            value_counts = df[col].value_counts().head(10)  # Топ-10 категорий
            value_counts.plot(kind='bar', ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Распределение {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Количество')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Скрываем пустые subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

# Основное приложение Streamlit
def main():
    st.set_page_config(page_title="Clustering Analysis App", layout="wide")
    st.title("🔍 Анализ данных и кластеризация")
    
    # Инициализация session_state для графиков
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
    
    # Инициализация для графиков предобработки
    if 'preprocessing_figures' not in st.session_state:
        st.session_state.preprocessing_figures = {
            'numeric_distributions': None,
            'categorical_distributions': None
        }
    if 'preprocessing_completed' not in st.session_state:
        st.session_state.preprocessing_completed = False
    
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
            
            # Показ предпросмотра данных
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head())

            # ВЫЗОВ ФУНКЦИИ show_info
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
                index=0
            )

            if selected_target:
                # Проверяем, является ли таргет числовым
                target_dtype = df[selected_target].dtype
                is_numeric = pd.api.types.is_numeric_dtype(df[selected_target])
                
                if is_numeric:
                    st.success(f"✅ Целевая переменная: **{selected_target}**")
                    st.info(f"Тип данных: {target_dtype}")
                    
                    # Показываем базовую статистику по таргету
                    target_stats = df[selected_target].describe()
                    st.write("**Базовая статистика:**")
                    st.write(f"- Минимум: {target_stats['min']:.2f}")
                    st.write(f"- Максимум: {target_stats['max']:.2f}")
                    st.write(f"- Среднее: {target_stats['mean']:.2f}")
                    st.write(f"- Медиана: {target_stats['50%']:.2f}")
                else:
                    st.error(f"❌ Целевая переменная '{selected_target}' должна быть числовой!")
                    st.warning(f"Текущий тип: {target_dtype}")
                    st.info("💡 Для преобразования в числовой формат используйте предобработку данных")

        with col1:
            st.subheader("Входные признаки")
            
            # Исключаем таргетную переменную из доступных для выбора признаков
            available_features = all_columns.copy()
            if selected_target and selected_target in available_features:
                available_features.remove(selected_target)
            
            # Автоматически выбираем признаки по умолчанию
            if len(available_features) > 0:
                default_features = available_features[:min(5, len(available_features))]
            else:
                default_features = []
            
            selected_features = st.multiselect(
                "Выберите признаки для анализа:",
                options=available_features,
                default=default_features,
                help="Таргетная переменная автоматически исключена из списка"
            )
            
            # Валидация
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
            remove_outliers = st.checkbox("Удалить выбросы", value=True)
        
        with preprocessing_col2:
            low_disp_threshold = st.slider(
                "Порог для удаления низкодисперсных категориальных признаков:",
                min_value=0.8, max_value=1.0, value=0.95, step=0.01
            )
            outlier_multiplier = st.slider(
                "Множитель для определения выбросов (IQR):",
                min_value=1.0, max_value=3.0, value=1.5, step=0.1
            )
        
        if st.button("Запустить предобработку"):
            if len(selected_features) == 0:
                st.error("❌ Нельзя запустить предобработку без выбранных признаков!")
            else:
                with st.spinner("Выполняется предобработка данных..."):
                    # Создаем копию данных с выбранными признаками
                    if selected_target and selected_target in selected_features:
                        features_to_use = [f for f in selected_features if f != selected_target]
                        st.warning(f"⚠️ Таргетная переменная '{selected_target}' исключена из входных признаков")
                    else:
                        features_to_use = selected_features.copy()
                    
                    if selected_target:
                        processed_df = df[[selected_target] + features_to_use].copy()
                    else:
                        processed_df = df[features_to_use].copy()
                    
                    # Определяем числовые и категориальные столбцы
                    num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if selected_target and selected_target in num_cols:
                        num_cols.remove(selected_target)
                    if selected_target and selected_target in cat_cols:
                        cat_cols.remove(selected_target)
                    
                    # Применяем очистку данных (всегда удаляем пропуски)
                    numeric_for_cleaning = num_cols.copy()
                    if selected_target and processed_df[selected_target].dtype in [np.number]:
                        numeric_for_cleaning.append(selected_target)
                    
                    processed_df = comprehensive_clean_data_cached(
                        processed_df,
                        remove_outliers=remove_outliers,
                        outlier_multiplier=outlier_multiplier,
                        numeric_columns=numeric_for_cleaning
                    )
                    
                    # Удаляем низкодисперсные категориальные признаки
                    if cat_cols:
                        processed_df = delete_cat_with_zero_dispersion(
                            processed_df, cat_cols, threshold=low_disp_threshold
                        )
                        # Обновляем список категориальных столбцов
                        cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
                        if selected_target and selected_target in cat_cols:
                            cat_cols.remove(selected_target)
                    
                    st.session_state.processed_df = processed_df
                    st.session_state.num_cols = num_cols
                    st.session_state.cat_cols = cat_cols
                    st.session_state.selected_target = selected_target
                    st.session_state.clustering_done = False  # Сбрасываем флаг кластеризации
                    st.session_state.tree_built = False  # Сбрасываем флаг дерева
                    st.session_state.preprocessing_completed = True
                    
                    st.success(f"✅ Предобработка завершена! Новый размер: {processed_df.shape}")
                    
                    # Показываем статистику после очистки
                    st.subheader("Статистика после очистки")
                    st.dataframe(processed_df.describe())
                    
                    # Визуализации распределений и сохраняем в session_state
                    if num_cols:
                        st.subheader("Распределения числовых признаков")
                        fig_num = plot_numeric_distributions(processed_df, num_cols)
                        st.session_state.preprocessing_figures['numeric_distributions'] = fig_num
                        st.pyplot(fig_num)
                    
                    if cat_cols:
                        st.subheader("Распределения категориальных признаков")
                        fig_cat = plot_categorical_distributions(processed_df, cat_cols)
                        st.session_state.preprocessing_figures['categorical_distributions'] = fig_cat
                        st.pyplot(fig_cat)
        
        # Показываем сохраненные графики предобработки, если они есть
        if st.session_state.preprocessing_completed:
            processed_df = st.session_state.processed_df
            num_cols = st.session_state.num_cols
            cat_cols = st.session_state.cat_cols
            
            # Всегда показываем статистику
            st.subheader("Статистика после очистки")
            st.dataframe(processed_df.describe())
            
            # Показываем сохраненные графики
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
                use_auto_clusters = st.checkbox("Автоматический выбор числа кластеров (elbow method)", value=True)
                
                if not use_auto_clusters:
                    n_clusters = st.slider("Число кластеров:", min_value=2, max_value=20, value=4)
                else:
                    n_clusters = None
            
            with cluster_col2:
                st.subheader("Информация о данных")
                st.write(f"Числовые признаки: {len(num_cols)}")
                st.write(f"Категориальные признаки: {len(cat_cols)}")
                st.write(f"Общее количество образцов: {len(processed_df)}")
            
            if st.button("Запустить кластеризацию"):
                with st.spinner("Выполняется кластеризация..."):
                    try:
                        # Подготовка данных для кластеризации
                        features_for_clustering = num_cols + cat_cols
                        X_processed, label_encoders, preprocessor = encoded_features_cached(
                            processed_df[features_for_clustering], num_cols, cat_cols
                        )
                        
                        # Дендрограмма признаков
                        st.subheader("Дендрограмма признаков")
                        dendro_fig = plot_dendogram_features(X_processed, features_for_clustering)
                        st.pyplot(dendro_fig)
                        
                        # Иерархическая кластеризация
                        Z = linkage(X_processed, method='ward', metric='euclidean')
                        
                        # Определение оптимального числа кластеров
                        if use_auto_clusters:
                            inertias = []
                            cluster_range = range(2, min(20, len(processed_df)))
                            
                            for k in cluster_range:
                                inertia = calculate_inertia(Z, k, X_processed)
                                inertias.append(inertia)
                            
                            optimal_clusters = find_elbow_point(inertias, cluster_range)
                            st.info(f"🎯 Автоматически найдено оптимальное число кластеров: {optimal_clusters}")
                        else:
                            optimal_clusters = n_clusters
                        
                        # Создание кластеров
                        clusters = fcluster(Z, optimal_clusters, criterion='maxclust')
                        clustered_df = processed_df.copy()
                        clustered_df['Cluster'] = clusters
                        
                        # Сохраняем все в session_state
                        st.session_state.clustered_df = clustered_df
                        st.session_state.optimal_clusters = optimal_clusters
                        st.session_state.X_processed = X_processed
                        st.session_state.clustering_done = True
                        st.session_state.tree_built = False  
                        
                        # Визуализация результатов
                        st.subheader("Результаты кластеризации")
                        
                        # Распределение по кластерам
                        cluster_dist = clustered_df['Cluster'].value_counts().sort_index()
                        fig_cluster_dist, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(cluster_dist.index, cluster_dist.values, 
                                    color='lightcoral', edgecolor='darkred', alpha=0.7)
                        
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
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_processed)
                        
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
                        
                        # Анализ характеристик кластеров
                        st.subheader("Анализ характеристик кластеров")
                        
                        if num_cols:
                            st.write("**Статистика числовых признаков по кластерам:**")
                            numeric_summary = clustered_df.groupby('Cluster')[num_cols].mean()
                            st.dataframe(numeric_summary.style.background_gradient(cmap='Blues'))
                        
                        if cat_cols:
                            st.write("**Распределение категориальных признаков по кластерам:**")
                            for cat_col in cat_cols:
                                cross_tab = pd.crosstab(clustered_df['Cluster'], 
                                                       clustered_df[cat_col], 
                                                       normalize='index') * 100
                                st.write(f"**{cat_col}:**")
                                st.dataframe(cross_tab.style.background_gradient(cmap='Blues', axis=0))
                        
                    except Exception as e:
                        st.error(f"Ошибка при кластеризации: {e}")
            
            # Показываем результаты кластеризации, если они уже есть
            if st.session_state.clustering_done:
                clustered_df = st.session_state.clustered_df
                optimal_clusters = st.session_state.optimal_clusters
                num_cols = st.session_state.num_cols
                cat_cols = st.session_state.cat_cols
                
                # Дерево решений для интерпретации кластеров
                st.header("5. Интерпретация кластеров с помощью дерева решений")
                
                if st.button("Построить дерево решений") or st.session_state.tree_built:
                    with st.spinner("Строим дерево решений..."):
                        try:
                            # Подготовка данных для дерева
                            X_for_tree = clustered_df[num_cols + cat_cols].copy()
                            y_clusters = clustered_df['Cluster']
                            
                            # One-Hot кодирование для категориальных признаков
                            X_encoded = pd.get_dummies(X_for_tree, columns=cat_cols, drop_first=True)
                            
                            # Разделение на train/test
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_encoded, y_clusters, test_size=0.3, random_state=42, stratify=y_clusters
                            )
                            
                            # Обучаем дерево решений
                            tree_clf = DecisionTreeClassifier(
                                max_depth=4,
                                min_samples_split=20,
                                min_samples_leaf=10,
                                random_state=42
                            )
                            
                            tree_clf.fit(X_train, y_train)
                            
                            # Визуализация дерева
                            st.subheader("Визуализация дерева решений")
                            fig_tree, ax = plt.subplots(figsize=(20, 12))
                            plot_tree(tree_clf,
                                     feature_names=X_encoded.columns.tolist(),
                                     class_names=[f'Cluster {i}' for i in range(1, optimal_clusters+1)],
                                     filled=True,
                                     rounded=True,
                                     fontsize=10,
                                     ax=ax)
                            ax.set_title('Дерево решений для предсказания кластеров', 
                                       fontsize=16, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig_tree)
                            
                            # Важность признаков
                            st.subheader("Важность признаков")
                            feature_importance = pd.DataFrame({
                                'feature': X_encoded.columns,
                                'importance': tree_clf.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            st.write("Топ-10 наиболее важных признаков:")
                            # Сбрасываем индекс и переименовываем столбец
                            feature_importance_reset = feature_importance.head(10).reset_index()
                            feature_importance_reset = feature_importance_reset.rename(columns={'index': '№'})
                            feature_importance_reset = feature_importance_reset.drop(columns='№')
                            st.dataframe(feature_importance_reset)
                            
                            # Визуализация важности признаков
                            fig_importance, ax = plt.subplots(figsize=(10, 6))
                            top_features = feature_importance.head(10)
                            top_features_sorted = top_features.sort_values('importance', ascending=True)
                            ax.barh(top_features_sorted['feature'], top_features_sorted['importance'])
                            ax.set_xlabel('Важность')
                            ax.set_title('Топ-10 наиболее важных признаков для определения кластеров')
                            plt.tight_layout()
                            st.pyplot(fig_importance)
                            
                            st.session_state.tree_built = True
                            
                        except Exception as e:
                            st.error(f"Ошибка при построении дерева: {e}")
            
            # Скачивание результатов
            if st.session_state.clustering_done:
                st.header("6. Скачивание результатов")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    st.session_state.clustered_df.to_excel(writer, sheet_name='Clustered_Data', index=False)
                    
                    # Статистика по кластерам
                    if st.session_state.num_cols or st.session_state.cat_cols:
                        cluster_stats = st.session_state.clustered_df.groupby('Cluster').agg({
                            **{col: 'mean' for col in st.session_state.num_cols},
                            **{col: lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A' for col in st.session_state.cat_cols}
                        })
                        cluster_stats.to_excel(writer, sheet_name='Cluster_Statistics')
                
                output.seek(0)
                
                st.download_button(
                    label="📥 Скачать результаты кластеризации (Excel)",
                    data=output,
                    file_name=f"clustering_results_{st.session_state.optimal_clusters}_clusters.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


if __name__ == "__main__":
    main()