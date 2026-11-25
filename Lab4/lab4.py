import os
from sys import exit
import matplotlib.pyplot as plt
from ts_analyzer import TimeSeriesAnalyzer
from ts_visualizer import TimeSeriesVisualizer
from ts_generator import SyntheticTSGenerator
from data_loader import load_psd_excel_series
from web_scraper import *

if __name__ == '__main__':

    directory_name = 'output'

    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    print('Оберіть напрям досліджень:')
    print('1 - Парсинг сайту обсерваторії (старий) https://cgo-sreznevskyi.kiev.ua/index.php?fn=k_klimat&f=kyiv')
    print('2 - Парсинг сайту обсерваторії (новий) https://cgo-sreznevskyi.kyiv.ua/uk/diialnist/klimatolohichna/klimatychni-dani-po-kyievu')
    print('3 - Зчитування інформації з файлу')
    mode = int(input('mode:'))

    if (mode == 1):
        print('Обрано інформаційне джерело: https://cgo-sreznevskyi.kiev.ua/index.php?fn=k_klimat&f=kyiv')
        url = 'https://cgo-sreznevskyi.kiev.ua/index.php?fn=k_klimat&f=kyiv'
        directory_name += '/szo'
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
        Parser_URL_szo_old(url)
        input_file = 'raw.csv'
        input_directory = directory_name + '/' + input_file

    elif (mode == 2):
        print('Обрано інформаційне джерело: https://cgo-sreznevskyi.kyiv.ua/uk/diialnist/klimatolohichna/klimatychni-dani-po-kyievu')
        url = 'https://cgo-sreznevskyi.kyiv.ua/uk/diialnist/klimatolohichna/klimatychni-dani-po-kyievu'
        directory_name += '/szo'
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
        Parser_URL_szo_new(url)
        input_file = 'raw.csv'
        input_directory = directory_name + '/' + input_file

    elif (mode == 3):
        input_file = input('Введіть назву файлу(формату .csv): ')
        directory_name += f'/{input_file}'
        print(directory_name)
        if not os.path.exists(directory_name):
            print('Ви ввели неправильну назву файлу')
            exit()
        else:
            input_directory = directory_name

    else:
        print('Неправильний варіант')
        exit()

    data = pd.read_csv(input_directory, index_col = 0)

    annual_temp = data['Середньорічна']

    analyzer = TimeSeriesAnalyzer(annual_temp, cycle_period=11)
    report = analyzer.run_full_analysis(n_clusters=3, nlags=15)

    print("=== 1. ВЛАСТИВОСТІ РЕАЛЬНОГО РЯДУ ===")
    print("Описова статистика:", report.descriptive)
    print("Стаціонарність:", report.stationarity)
    print(f"Сила тренду F_T = {report.trend_strength:.3f}, Сила циклу F_S = {report.seasonal_strength:.3f}")
    print(f"Silhouette кластеризації = {report.silhouette:.3f}")

    generator = SyntheticTSGenerator(trend_degree=2, cycle_period=11.0).fit(annual_temp.values)
    sim_matrix = generator.generate(n_steps=len(annual_temp), n_simulations=500, seed=42)
    metrics = generator.validate_adequacy(annual_temp.values, sim_matrix)

    print("\n=== 2. АДЕКВАТНІСТЬ СИНТЕТИЧНИХ МОДЕЛЕЙ ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    t = np.arange(len(annual_temp), dtype=float)
    poly_trend = np.polyval(generator.poly_coeffs, t)

    vis = TimeSeriesVisualizer()
    cluster_labels = [
        "Кластер 0: Холодний режим",
        "Кластер 1: Помірний режим",
        "Кластер 2: Теплий режим",
    ]

    vis.plot_decomposition(
        ts=annual_temp,
        trend=report.trend,
        seasonal=report.seasonal,
        resid=report.resid,
        poly_trend=poly_trend,
        title="Декомпозиція середньорічної температури (1881–2025, цикл = 11 років)",
        ylabel="T (°C)",
        save_path="climate_1_decomposition.png",
    )

    vis.plot_clusters_and_correlations(
        ts=annual_temp,
        clusters=report.clusters,
        acf_raw=report.acf_raw,
        pacf_raw=report.pacf_raw,
        acf_detrended=report.acf_detrended,
        pacf_detrended=report.pacf_detrended,
        cluster_names=cluster_labels,
        ylabel="T (°C)",
        save_path="climate_2_clusters_corr.png",
    )

    vis.plot_synthetic_validation(
        ts=annual_temp,
        sim_matrix=sim_matrix,
        metrics=metrics,
        ylabel="T (°C)",
        save_path="climate_3_synthetic_validation.png",
    )

    vis.plot_summary_dashboard(
        ts=annual_temp,
        trend=report.trend,
        clusters=report.clusters,
        acf_raw=report.acf_raw[:13],
        pacf_raw=report.pacf_raw[:13],
        sim_matrix=sim_matrix,
        metrics=metrics,
        dataset_title="Середньорічна температура (raw.csv)",
        ylabel="T (°C)",
        cluster_names=cluster_labels,
        save_path="climate_4_dashboard.png",
    )
    plt.show()

    ts = load_psd_excel_series("Data_Set_4.xlsx", sheet_name="1 Type of sale", metric_code="1.1")
    ts_k = ts / 1000.0  # Переводимо в тисячі угод для зручності підписів на графіках

    analyzer = TimeSeriesAnalyzer(ts_k, cycle_period=4)
    report = analyzer.run_full_analysis(n_clusters=3, nlags=12)

    gen = SyntheticTSGenerator(trend_degree=3, cycle_period=4.0).fit(np.log(ts_k.values))
    sim_matrix = np.exp(gen.generate(n_steps=len(ts_k), n_simulations=500, seed=42))
    metrics = gen.validate_adequacy(ts_k.values, sim_matrix)

    vis = TimeSeriesVisualizer()
    cluster_labels = [
        "Кластер 0: Посткризова стагнація",
        "Кластер 1: Кризовий спад (2008)",
        "Кластер 2: Докризовий бум (2005–2007)",
    ]

    vis.plot_decomposition(
        ts=ts_k,
        trend=report.trend,
        seasonal=report.seasonal,
        resid=report.resid,
        title="Декомпозиція іпотечних продажів (2005-Q2 – 2014-Q1, сезон = 4 кв.)",
        ylabel="Угоди (тис.)",
        save_path="mortgage_1_decomposition.png",
    )

    vis.plot_clusters_and_correlations(
        ts=ts_k,
        clusters=report.clusters,
        acf_raw=report.acf_raw,
        pacf_raw=report.pacf_raw,
        acf_detrended=report.acf_detrended,
        pacf_detrended=report.pacf_detrended,
        cluster_names=cluster_labels,
        ylabel="Угоди (тис.)",
        save_path="mortgage_2_clusters_corr.png",
    )

    vis.plot_synthetic_validation(
        ts=ts_k,
        sim_matrix=sim_matrix,
        metrics=metrics,
        ylabel="Угоди (тис.)",
        save_path="mortgage_3_synthetic_validation.png",
    )

    vis.plot_summary_dashboard(
        ts=ts_k,
        trend=report.trend,
        clusters=report.clusters,
        acf_raw=report.acf_raw,
        pacf_raw=report.pacf_raw,
        sim_matrix=sim_matrix,
        metrics=metrics,
        dataset_title="Іпотечні продажі (Data_Set_4.xlsx)",
        ylabel="Угоди (тис.)",
        cluster_names=cluster_labels,
        save_path="mortgage_4_dashboard.png",
    )
    plt.show()