import os
from sys import exit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from web_scraper import *

roman_to_num = {
    'I': '01', 
    'II': '02', 
    'III': '03', 
    'IV': '04', 
    'V': '05', 
    'VI': '06',
    'VII': '07', 
    'VIII': '08', 
    'IX': '09', 
    'X': '10', 
    'XI': '11', 
    'XII': '12'
}

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
    years = data.index
    y_raw = annual_temp.values

    n = len(y_raw)
    x = np.arange(n, dtype=float)  # t = Рік - 1881

    # 2. Очищення від аномальних вимірів (фільтр Хампеля)
    window = 15
    k_sigma = 3.0
    rolling_med = pd.Series(y_raw).rolling(window=window, center=True, min_periods=1).median().values
    mad = np.median(np.abs(y_raw - rolling_med))
    sigma_robust = 1.4826 * mad
    z_hampel = np.abs(y_raw - rolling_med) / sigma_robust
    outliers_idx = np.where(z_hampel > k_sigma)[0]

    y_clean = y_raw.copy()
    y_clean[outliers_idx] = rolling_med[outliers_idx]

    # 3. Реалізація МНК через QR-розклад
    def fit_ols_qr(x_vec: np.ndarray, y_vec: np.ndarray, deg: int) -> np.ndarray:
        X = np.vander(x_vec, deg + 1, increasing=True)
        Q, R = np.linalg.qr(X)
        return np.linalg.solve(R, Q.T @ y_vec)

    def predict_poly(x_vec: np.ndarray, beta: np.ndarray) -> np.ndarray:
        X = np.vander(x_vec, len(beta), increasing=True)
        return X @ beta

    # 4. Оптимізація степеня полінома за критерієм BIC
    results = []
    for d in range(1, 7):
        beta = fit_ols_qr(x, y_clean, d)
        y_hat = predict_poly(x, beta)
        rss = np.sum((y_clean - y_hat) ** 2)
        rmse = np.sqrt(rss / n)
        r2 = 1.0 - rss / np.sum((y_clean - np.mean(y_clean)) ** 2)
        k = d + 1
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        results.append({'d': d, 'RSS': rss, 'RMSE': rmse, 'R2': r2, 'AIC': aic, 'BIC': bic, 'beta': beta})

    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['BIC'].idxmin()]
    best_d = int(best_row['d'])
    best_beta = best_row['beta']

    # 5. Верифікація скрипта
    beta_polyfit = np.polyfit(x, y_clean, best_d)[::-1]
    X_best = np.vander(x, best_d + 1, increasing=True)
    residuals = y_clean - predict_poly(x, best_beta)

    assert np.allclose(best_beta, beta_polyfit, atol=1e-12), "Помилка верифікації коефіцієнтів!"
    assert np.linalg.norm(X_best.T @ residuals) < 1e-5, "Умова ортогональності залишків порушена!"

    # 6. Прогнозування (екстраполяція) на 0.5 інтервалу спостереження (72 роки)
    horizon = int(np.floor(0.5 * n))
    x_fut = np.arange(n, n + horizon, dtype=float)
    years_fut = np.arange(years[-1] + 1, years[-1] + 1 + horizon)
    y_fut = predict_poly(x_fut, best_beta)

    # Розрахунок 95% інтервалу прогнозу
    s2 = np.sum(residuals ** 2) / (n - (best_d + 1))
    XtX_inv = np.linalg.inv(X_best.T @ X_best)
    X_fut = np.vander(x_fut, best_d + 1, increasing=True)
    pred_se = np.sqrt(s2 * (1.0 + np.sum((X_fut @ XtX_inv) * X_fut, axis=1)))
    t_crit = stats.t.ppf(0.975, df=n - (best_d + 1))

    ci_low = y_fut - t_crit * pred_se
    ci_high = y_fut + t_crit * pred_se

    print(f"Оптимальний степінь: d = {best_d}, BIC = {best_row['BIC']:.2f}, R2 = {best_row['R2']:.4f}")
    for yr, val, lo, hi in zip(years_fut[[0, 24, -1]], y_fut[[0, 24, -1]], ci_low[[0, 24, -1]], ci_high[[0, 24, -1]]):
        print(f"Прогноз на {yr} р.: {val:.2f} °C (95% ПІ: [{lo:.2f}; {hi:.2f}])")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Графік 1: Очищення вхідних даних від аномалій (фільтр Хампеля)
    ax1 = axes[0, 0]
    ax1.plot(years, y_raw, color='gray', alpha=0.6, label='Вхідні дані (raw)')
    ax1.plot(years, rolling_med, color='tab:blue', linestyle='--', linewidth=1.5, label=f'Ковзна медіана (w={window})')
    ax1.scatter(years[outliers_idx], y_raw[outliers_idx], color='red', s=70, zorder=5,
                label=f'Аномалія: {years[outliers_idx[0]]} р. ({y_raw[outliers_idx[0]]:.1f} °C)')
    ax1.scatter(years[outliers_idx], y_clean[outliers_idx], color='green', s=70, zorder=5,
                label=f'Заміна: {y_clean[outliers_idx[0]]:.1f} °C')
    ax1.set_title('1. Очищення даних від аномалій (фільтр Хампеля)')
    ax1.set_xlabel('Рік')
    ax1.set_ylabel('Температура, °C')
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Графік 2: Оптимізація степеня полінома за інформаційними критеріями AIC та BIC
    ax2 = axes[0, 1]
    ax2.plot(res_df['d'], res_df['BIC'], marker='o', linewidth=2, color='tab:purple', label=f'BIC (мінімум при d={best_d})')
    ax2.plot(res_df['d'], res_df['AIC'], marker='s', linewidth=2, linestyle='--', color='tab:orange', label='AIC')
    ax2.axvline(best_d, color='red', linestyle=':', label=f'Оптимальний степінь d={best_d}')
    ax2.set_title('2. Вибір оптимального степеня полінома (AIC / BIC)')
    ax2.set_xlabel('Степінь полінома (d)')
    ax2.set_ylabel('Значення критерію')
    ax2.set_xticks(range(1, 7))
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # Графік 3: Апроксимація на навчальній вибірці та екстраполяція на 0.5 інтервалу
    ax3 = axes[1, 0]
    ax3.scatter(years, y_clean, color='steelblue', alpha=0.55, s=18, label=f'Очищені дані ({years[0]}–{years[-1]})')
    ax3.plot(years, predict_poly(x, best_beta), color='darkblue', linewidth=2.2, label=f'МНК-поліном (d={best_d})')
    ax3.plot(years_fut, y_fut, color='crimson', linewidth=2.2, linestyle='--',
            label=f'Екстраполяція ({years_fut[0]}–{years_fut[-1]}, {horizon} р.)')
    ax3.fill_between(years_fut, ci_low, ci_high, color='crimson', alpha=0.15, label='95% інтервал прогнозу')
    ax3.axvline(years[-1], color='black', linestyle=':', alpha=0.7)
    ax3.set_title('3. Навчання МНК-моделі та екстраполяція на 0.5 вибірки')
    ax3.set_xlabel('Рік')
    ax3.set_ylabel('Середньорічна температура, °C')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Графік 4: Аналіз залишків для верифікації адекватності моделі
    ax4 = axes[1, 1]
    ax4.scatter(years, residuals, color='teal', alpha=0.65, s=20, label='Залишки моделі (e = y - ŷ)')
    ax4.axhline(0, color='black', linewidth=1.2)
    ax4.axhline(2 * np.sqrt(s2), color='red', linestyle='--', alpha=0.6, label='±2σ залишків')
    ax4.axhline(-2 * np.sqrt(s2), color='red', linestyle='--', alpha=0.6)
    ax4.set_title('4. Залишки моделі (валідація адекватності)')
    ax4.set_xlabel('Рік')
    ax4.set_ylabel('Залишок e = y - ŷ, °C')
    ax4.legend(fontsize=8.5)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()