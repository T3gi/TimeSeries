import os
from sys import exit
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
    print(data)

    annual_temp = data['Середньорічна']

    x = data.index.values
    y = annual_temp.values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    trend_start = intercept + slope * x.min()
    trend_end = intercept + slope * x.max()
    total_trend_change = trend_end - trend_start
    rolling_10 = annual_temp.rolling(window = 10, center = True).mean()

    print("\nОЦІНКА ДИНАМІКИ ТРЕНДУ")
    print(f"Період спостережень: {x.min()} – {x.max()} рр. ({len(data)} років)")
    print(f"Середня швидкість зміни (кут нахилу): {slope:+.4f} °C/рік")
    print(f"Темп потепління за декаду: {slope * 10:+.4f} °C/10 років")
    print(f"Початкове розрахункове значення тренду ({x.min()} р.): {trend_start:.2f} °C")
    print(f"Кінцеве розрахункове значення тренду ({x.max()} р.): {trend_end:.2f} °C")
    print(f"Сумарна зміна за період згідно з трендом: {total_trend_change:+.2f} °C")
    print(f"Коефіцієнт детермінації (R²): {r_value**2:.4f}")
    print(f"Статистична значущість (p-value): {p_value:.2e}")

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, annual_temp.values, label='Фактичні дані', color='steelblue', alpha=0.6, linewidth=1.2)
    plt.plot(data.index, intercept + slope * x, label=f'Лінійний тренд ({slope*10:+.2f} °C / 10 р.)', color='crimson', linewidth=2)
    plt.plot(data.index, rolling_10, label='10-річне ковзне середнє', color='darkorange', linewidth=2, linestyle='--')

    plt.title('Динаміка середньорічної температури (1881–2018)', fontsize=14, fontweight='bold')
    plt.xlabel('Рік', fontsize=12)
    plt.ylabel('Температура (°C)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()

    stats_summary = annual_temp.describe()

    additional_stats = pd.Series(
        {
            "Дисперсія (var)": annual_temp.var(),
            "Медіана (median)": annual_temp.median(),
            "Розмах (range)": annual_temp.max() - annual_temp.min(),
            "Міжквартильний розмах (IQR)": annual_temp.quantile(0.75)
            - annual_temp.quantile(0.25),
            "Асиметрія (skewness)": annual_temp.skew(),
            "Ексцес (kurtosis)": annual_temp.kurt(),
        }
    )

    print("\nОсновні статистичні характеристики")
    print(stats_summary)
    print(additional_stats)

    poly_coeffs = np.polyfit(x, y, deg=2)
    trend = np.polyval(poly_coeffs, x)

    residuals = y - trend
    mean_res = np.mean(residuals)
    std_res = np.std(residuals, ddof=1)

    phi_1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    sigma_noise = np.sqrt(std_res**2 * (1 - phi_1**2))

    np.random.seed(42)  
    n_samples = len(x)

    synthetic_residuals = np.zeros(n_samples)
    synthetic_residuals[0] = np.random.normal(0, std_res)

    for t in range(1, n_samples):
        noise = np.random.normal(0, sigma_noise)
        synthetic_residuals[t] = phi_1 * synthetic_residuals[t - 1] + noise

    synthetic_temp = trend + synthetic_residuals

    synthetic_df = pd.DataFrame({
        'Рік': x,
        'Середньорічна_реальна': y,
        'Середньорічна_синтетична': np.round(synthetic_temp, 1)
    }).set_index('Рік')

    comparison = pd.DataFrame({
        'Показник': ['Середнє', 'Станд. відхилення', 'Мінімум', 'Максимум'],
        'Реальні': [
            y.mean(),
            y.std(),
            y.min(),
            y.max()
        ],
        'Синтетичні': [
            synthetic_temp.mean(),
            synthetic_temp.std(),
            synthetic_temp.min(),
            synthetic_temp.max()
        ]
    })

    print("\nПорівняння статистичних характеристик:")
    print(comparison.to_string(index=False))

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, label='Реальні дані', color='tab:blue', alpha=0.7)
    plt.plot(x, trend, label='Квадратичний тренд', color='red', linestyle='--')
    plt.plot(x, synthetic_temp, label='Синтезована модель', color='tab:orange', alpha=0.8)
    plt.title('Порівняння реальних та синтезованих даних середньорічної температури')
    plt.xlabel('Рік')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()