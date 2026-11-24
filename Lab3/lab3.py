import os
from sys import exit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
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
    years = data.index
    y_raw = annual_temp.values

    def hampel_compensation(series: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0):
        n = len(series)
        cleaned = series.copy()
        anomalies = []
        half_w = window_size // 2
        k_gauss = 1.4826  

        for i in range(n):
            window = series[max(0, i - half_w):min(n, i + half_w + 1)]
            med = np.median(window)
            mad = k_gauss * np.median(np.abs(window - med))
            if mad > 0 and np.abs(series[i] - med) > n_sigmas * mad:
                z_mad = (series[i] - med) / mad
                anomalies.append((int(years[i]), series[i], med, z_mad))
                cleaned[i] = med
        return cleaned, anomalies

    y_clean, detected_anomalies = hampel_compensation(y_raw, window_size=11, n_sigmas=3.0)

    d2 = np.diff(y_clean, n=2)
    sigma_v = np.sqrt(np.mean(d2**2) / 6.0)
    d3 = np.diff(y_clean, n=3)
    sigma_w = np.sqrt(max(1e-6, np.var(d3) - 20.0 * sigma_v**2))
    lam = sigma_w / sigma_v  

    def init_state(z: np.ndarray, m: int = 5):
        """Ініціалізація початкового стану за МНК на перших m точках для усунення перехідного кидка."""
        coeffs = np.polyfit(np.arange(m), z[:m], 1)
        return coeffs[1], coeffs[0], 0.0

    def rls_diverging(z: np.ndarray, dt: float = 1.0):
        """Класичний рекурентний МНК / Калман без шуму процесу (демонструє розходження)."""
        n = len(z)
        x_hat = np.zeros(n)
        x, v, _ = init_state(z)
        for k in range(n):
            kk = k + 1
            alpha_k = 2.0 * (2.0 * kk - 1.0) / (kk * (kk + 1.0))
            beta_k = 6.0 / (kk * (kk + 1.0) * dt)
            x_pred = x + v * dt
            r = z[k] - x_pred
            x = x_pred + alpha_k * r
            v = v + (beta_k / dt) * r
            x_hat[k] = x
        return x_hat

    def alpha_beta_filter(z: np.ndarray, theta: float = 0.85, dt: float = 1.0):
        """Скалярний alpha-beta фільтр зі згасаючою пам'яттю."""
        alpha = 1.0 - theta**2
        beta = (1.0 - theta)**2
        n = len(z)
        x_hat, innov = np.zeros(n), np.zeros(n)
        x, v, _ = init_state(z)
        for k in range(n):
            x_pred = x + v * dt
            r = z[k] - x_pred
            innov[k] = r
            x = x_pred + alpha * r
            v = v + (beta / dt) * r
            x_hat[k] = x
        return x_hat, innov

    def alpha_beta_gamma_filter(z: np.ndarray, theta: float = 0.85, dt: float = 1.0):
        """Скалярний alpha-beta-gamma фільтр 3-го порядку (незміщений для квадратичного тренду)."""
        alpha = 1.0 - theta**3
        beta = 1.5 * (1.0 - theta)**2 * (1.0 + theta)
        gamma = 0.5 * (1.0 - theta)**3
        n = len(z)
        x_hat, innov = np.zeros(n), np.zeros(n)
        x, v, a = init_state(z)
        for k in range(n):
            x_pred = x + v * dt + 0.5 * a * dt**2
            v_pred = v + a * dt
            r = z[k] - x_pred
            innov[k] = r
            x = x_pred + alpha * r
            v = v_pred + (beta / dt) * r
            a = a + (2.0 * gamma / dt**2) * r
            x_hat[k] = x
        return x_hat, innov

    def fourth_order_abgd_filter(z: np.ndarray, theta: float = 0.85, dt: float = 1.0):
        """Модифікація 2.2: Підвищення ступеня фільтра до 4-го порядку (alpha-beta-gamma-delta)."""
        alpha = 1.0 - theta**4
        beta = (1.0 / 6.0) * (1.0 - theta)**2 * (11.0 + 14.0 * theta + 11.0 * theta**2)
        gamma = (1.0 - theta)**3 * (1.0 + theta)
        delta = (1.0 / 6.0) * (1.0 - theta)**4
        n = len(z)
        x_hat, innov = np.zeros(n), np.zeros(n)
        x, v, a = init_state(z)
        j = 0.0
        for k in range(n):
            x_pred = x + v * dt + 0.5 * a * dt**2 + (1.0 / 6.0) * j * dt**3
            v_pred = v + a * dt + 0.5 * j * dt**2
            a_pred = a + j * dt
            r = z[k] - x_pred
            innov[k] = r
            x = x_pred + alpha * r
            v = v_pred + (beta / dt) * r
            a = a_pred + (2.0 * gamma / dt**2) * r
            j = j + (6.0 * delta / dt**3) * r
            x_hat[k] = x
        return x_hat, innov

    def adaptive_abg_filter(z: np.ndarray, sigma_v_est: float, theta_min: float = 0.72,
                            theta_max: float = 0.92, rho: float = 0.15, dt: float = 1.0):
        """Модифікація 2.3: Адаптивний alpha-beta-gamma фільтр зі змінною пам'яттю за NIS."""
        n = len(z)
        x_hat, innov, thetas = np.zeros(n), np.zeros(n), np.zeros(n)
        x, v, a = init_state(z)
        S_k = sigma_v_est**2

        for k in range(n):
            x_pred = x + v * dt + 0.5 * a * dt**2
            v_pred = v + a * dt
            r = z[k] - x_pred
            innov[k] = r

            S_k = (1.0 - rho) * S_k + rho * (r**2)
            nis = (r**2) / max(S_k, 1e-6)
            theta_k = theta_min + (theta_max - theta_min) * np.exp(-0.5 * max(0.0, nis - 1.0))
            thetas[k] = theta_k

            alpha = 1.0 - theta_k**3
            beta = 1.5 * (1.0 - theta_k)**2 * (1.0 + theta_k)
            gamma = 0.5 * (1.0 - theta_k)**3

            x = x_pred + alpha * r
            v = v_pred + (beta / dt) * r
            a = a + (2.0 * gamma / dt**2) * r
            x_hat[k] = x
        return x_hat, innov, thetas

    x_rls = rls_diverging(y_clean)
    x_ab, inv_ab = alpha_beta_filter(y_clean, theta=0.85)
    x_abg, inv_abg = alpha_beta_gamma_filter(y_clean, theta=0.85)
    x_abgd, inv_abgd = fourth_order_abgd_filter(y_clean, theta=0.85)
    x_adap, inv_adap, thetas = adaptive_abg_filter(y_clean, sigma_v_est=sigma_v)

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    ax = axes[0]
    ax.plot(years, y_raw, color='#95a5a6', alpha=0.75, linewidth=1.1, label='Вихідний ряд (raw.csv)')
    ax.plot(years, y_clean, color='#2980b9', linewidth=1.5, label='Очищений ряд (компенсація Хемпеля)')

    anom_yrs = [a[0] for a in detected_anomalies]
    anom_vals = [a[1] for a in detected_anomalies]
    anom_meds = [a[2] for a in detected_anomalies]

    ax.scatter(anom_yrs, anom_vals, color='#e74c3c', s=75, zorder=5, label='Виявлені аномалії (|z_MAD| > 3)')
    ax.scatter(anom_yrs, anom_meds, color='#27ae60', marker='s', s=55, zorder=5, label='Компенсовані значення (локальна медіана)')
    for yr, val in zip(anom_yrs, anom_vals):
        ax.annotate(
            f"{yr} ({val}°C)",
            (yr, val),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=9,
            color='#c0392b',
            fontweight='bold'
        )

    ax.set_title('1. Дослідження вхідних даних та компенсація аномалій (Фільтр Хемпеля, вікно W=11)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Температура, °C')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9)

    ax = axes[1]
    t_idx = np.arange(len(y_clean))
    p3_trend = np.polyval(np.polyfit(t_idx, y_clean, 3), t_idx)

    theta_std = 0.85
    a_ab, b_ab = 1.0 - theta_std**2, (1.0 - theta_std)**2
    a_abg = 1.0 - theta_std**3
    b_abg = 1.5 * (1.0 - theta_std)**2 * (1.0 + theta_std)
    g_abg = 0.5 * (1.0 - theta_std)**3

    ax.scatter(years, y_clean, color='#bdc3c7', s=14, alpha=0.7, label='Очищені спостереження')
    ax.plot(years, p3_trend, color='black', linestyle=':', linewidth=1.5, label='Поліноміальний тренд (3-й ступінь)')
    ax.plot(years, x_rls, color='#e74c3c', linestyle='--', linewidth=1.8, label='Рекурентний МНК / Калман без Q (Ефект розходження)')
    ax.plot(years, x_ab, color='#f39c12', linewidth=1.8, label=f'α-β фільтр (θ={theta_std}; α={a_ab:.3f}, β={b_ab:.4f})')
    ax.plot(years, x_abg, color='#16a085', linewidth=2.2, label=f'α-β-γ фільтр (θ={theta_std}; α={a_abg:.3f}, β={b_abg:.4f}, γ={g_abg:.4f})')

    ax.set_title('2. Рекурентне згладжування Калмана: усунення зміщення у α-β-γ фільтрі проти розходження МНК', fontsize=11, fontweight='bold')
    ax.set_ylabel('Температура, °C')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9)

    ax = axes[2]
    ax.plot(years, y_clean, color='#bdc3c7', linewidth=1.0, alpha=0.6, label='Очищені дані')
    ax.plot(years, x_abg, color='#16a085', linewidth=1.8, label='3-й порядок: α-β-γ (згасаюча пам’ять θ=0.85)')
    ax.plot(years, x_abgd, color='#8e44ad', linestyle='--', linewidth=1.6, label='4-й порядок: α-β-γ-δ (підвищення ступеня)')
    ax.plot(years, x_adap, color='#2c3e50', linewidth=2.0, label='Адаптивний α-β-γ фільтр (змінна пам’ять θ_k)')

    ax.set_title('3. Група вимог 2: Модифікації фільтра та динаміка адаптивного параметра пам’яті θ_k', fontsize=11, fontweight='bold')
    ax.set_xlabel('Рік спостережень')
    ax.set_ylabel('Температура, °C')
    ax.grid(True, linestyle='--', alpha=0.4)

    ax_twin = ax.twinx()
    ax_twin.plot(years, thetas, color='#e67e22', alpha=0.55, linewidth=1.3, label='Параметр пам’яті θ_k (права вісь)')
    ax_twin.set_ylabel('Коефіцієнт пам’яті θ_k', color='#d35400')
    ax_twin.set_ylim(0.65, 1.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.5)

    plt.tight_layout()
    plt.show()